import torch
from torch import nn, Tensor
from scipy.optimize import linprog

class OTLoss(nn.Module):
    """
    This class represents the Optimal Transport transportation cost and is used to calcualate the loss value.
    """
    def __init__(self):
        super(OTLoss, self).__init__()

    def forward(self, source: Tensor, target: Tensor, plan: Tensor, cost: Tensor) -> Tensor:
        """
        Args:
            source (Tensor): Tensor of token embeddings of the source sentence.
            target (Tensor): Tensor of token embeddings of the target sentence.
            plan (Tensor): The transportation plan.
            cost (Tensor): The transportation cost.
        Returns:
            Tensor: Total loss as transportation cost.
        """
        loss: Tensor = torch.sum(plan * cost)
        return loss

class OTSolver(nn.Module):
    """
    This class is used to solve the Optimal Transport problem.
    """
    def __init__(
        self, 
        device: str="cpu",
        epsilon: float=1e-3, 
        beta: float=0.5,       # Increased from 2 → 0.5: exp(-C/beta) less extreme, prevents Q→0
        max_iter: int=100,     # Reduced from 1000: each outer iter is now better converged
        L: int=5,              # Increased from 1: more inner Sinkhorn steps per outer iter
        use_path: bool=True, 
        tol: float=1e-9, 
        method: str='ipot'
    ) -> None:
        """
        Args:
            beta (float): Step size of proximal point iteration.
            L (int): Number of iterations for inner optimization.
            use_path (bool): Whether warm start method is used.
        """
        super(OTSolver, self).__init__()

        self.device = device
        self.epsilon: float = epsilon
        self.beta = beta
        self.max_iter: int = max_iter
        self.L = L
        self.use_path = use_path
        self.tol: float = tol
        self.method: str = method
        self.ot_loss: OTLoss = OTLoss().to(device)

    def forward(self, mu: Tensor, nu: Tensor, C: Tensor) -> tuple[Tensor, Tensor]:
        """
        Solves the Optimal Transport problem using the specified method.

        Args:
            mu (Tensor): Source distribution.
            nu (Tensor): Target distribution.
            C (Tensor): Cost matrix.
        Returns:
            Tensor: Transport plan.
        """
        if self.method == 'sinkhorn':
            plan = self.sinkhorn_knopp(mu, nu, C)
        elif self.method == 'ipot':
            plan = self.ipot(mu, nu, C)
        else:
            plan = self.linear_programming(mu, nu, C)
        loss = self.ot_loss(mu, nu, plan, C)
        return plan, loss

    def sinkhorn_knopp(self, mu: Tensor, nu: Tensor, C: Tensor) -> Tensor:
        """
        Solves the Optimal Transport problem using the Sinkhorn-Knopp algorithm.

        Args:
            mu (Tensor): Source distribution.
            nu (Tensor): Target distribution.
            C (Tensor): Cost matrix.
        Returns:
            Tensor: Transport plan.
        """
        K = torch.exp(-C / self.epsilon)  # Kernel matrix using entropy regularization
        u = torch.ones_like(mu, device=self.device)
        v = torch.ones_like(nu, device=self.device)
        
        for _ in range(self.max_iter):
            u_new = mu / (torch.matmul(K, v) + self.tol)  # Adding tolerance for numerical stability
            v_new = nu / (torch.matmul(K.T, u_new) + self.tol)
            
            # Check for convergence
            if torch.norm(u_new - u) < self.tol and torch.norm(v_new - v) < self.tol:
                break
            
            u, v = u_new, v_new
        
        # Transport plan is the element-wise multiplication of K, u, and v
        P = torch.diag(u) @ K @ torch.diag(v)
        
        return P
        
    def ipot(self, mu: Tensor, nu: Tensor, C: Tensor) -> Tensor:
        """
        Inexact Proximal Point (IPOT) algorithm for Optimal Transport.
        
        Args:
            mu (Tensor): Source distribution.
            nu (Tensor): Target distribution.
            C (Tensor): Cost matrix.
        Returns:
            Tensor: Transport plan.
        """
        m = len(mu)
        n = len(nu)
        a = torch.ones([m,], requires_grad=True, device=self.device)
        b = torch.ones([n,], requires_grad=True, device=self.device)

        Gamma = torch.ones((m, n), requires_grad=True, device=self.device) / (m * n)
        G = torch.exp(-(C / self.beta))

        for _ in range(self.max_iter):
            Q = G * Gamma
            if not self.use_path:
                a = torch.ones([m,], requires_grad=True, device=self.device)
                b = torch.ones([n,], requires_grad=True, device=self.device)
            
            for i in range(self.L):
                # Clamp denominator to avoid division by zero → NaN
                denom_a = torch.clamp(torch.matmul(Q, b), min=1e-8)
                a = mu / denom_a
                denom_b = torch.clamp(torch.matmul(Q.t(), a), min=1e-8)
                b = nu / denom_b
        
            Gamma = a.unsqueeze(1) * Q * b.unsqueeze(0)
                
        return Gamma
    
    def linear_programming(self, mu: Tensor, nu: Tensor, C: Tensor) -> Tensor:
        """
        Solves the Optimal Transport problem using linear programming.

        Args:
            mu (Tensor): Source distribution.
            nu (Tensor): Target distribution.
            C (Tensor): Cost matrix.
        Returns:
            Tensor: Transport plan.
        """
        n, m = C.shape
        c = C.flatten().cpu().numpy()  

        # Construct equality constraints
        A_eq = torch.zeros((n + m, n * m))
        for i in range(n):
            A_eq[i, i * m:(i + 1) * m] = 1  # Row constraints for source distribution
        for j in range(m):
            A_eq[n + j, j::m] = 1  # Column constraints for target distribution
        
        b_eq = torch.cat([mu, nu])  

        # Convert to NumPy for scipy.linprog
        A_eq_np = A_eq.numpy()
        b_eq_np = b_eq.cpu().numpy()

        # Bounds for each variable (non-negative transport plan)
        bounds = [(0, None) for _ in range(n * m)]

        result = linprog(c, A_eq=A_eq_np, b_eq=b_eq_np, bounds=bounds, method='highs')

        if not result.success:
            raise ValueError(f"Linear programming failed: {result.message}")

        P = torch.tensor(result.x, device=self.device).reshape(n, m)
        return P