import numpy as np
import ot
import matplotlib.pyplot as plt

class IPOT:
    def __init__(self, a1, a2, beta=2, max_iter=1000, L=1, use_path = True, return_map = True, return_loss = True):
        """
        Solve the optimal transport problem and return the OT matrix

        Parameters
        ----------
        a1 : np.ndarray (ns,ds)
            samples weights in the source domain
        a2 : np.ndarray (nt,dt)
            samples in the target domain
        C : np.ndarray (ns,nt)
            loss matrix
        beta : float, optional
            Step size of poximal point iteration
        max_iter : int, optional
            Max number of iterations
        L : int, optional
            Number of iterations for inner optimization
        use_path : bool, optional
            Whether warm start method is used
        return_map : bool, optional
            Whether the optimal transportation map is returned
        return_loss : bool, optional
            Whether the list of calculated WD is returned
        """
        self.a1 = a1
        self.a2 = a2
        self.beta = beta
        self.max_iter = max_iter
        self.L = L
        self.use_path = use_path
        self.return_map = return_map
        self.return_loss = return_loss

    def run(self):
        """
        Returns
        -------
        gamma : (ns x nt) ndarray
            Optimal transportation matrix for the given parameters
        loss : list
            log of loss (Wasserstein distance)
        """

        C = ot.dist(self.a1, self.a2, metric='euclidean')
        C /= C.max()
        distr_a1 = np.ones(len(self.a1))/len(self.a1) 
        distr_a2 = np.ones(len(self.a2))/len(self.a2)

        P, loss = self.run_IPOT(distr_a1, distr_a2, C, self.beta, self.max_iter, self.L, self.use_path, self.return_map, self.return_loss)
        return P, loss

    def run_IPOT(self, a1, a2, C, beta, max_iter, L, use_path, return_map, return_loss):
        m = len(a1)
        n = len(a2)
        a = np.ones([m,])
        b = np.ones([n,])

        Gamma = np.ones((m,n))/m*n
        G = np.exp(-(C/beta))

        if return_loss==True:
            loss = []

        for _ in range(max_iter):
            Q = G*Gamma
            if use_path == False:
                a = np.ones([m,])
                b = np.ones([n,])
            
            for i in range(L):
                a = a1/np.matmul(Q,b)
                b = a2/np.matmul(np.transpose(Q),a)
        
            Gamma = np.expand_dims(a,axis=1) * Q * np.expand_dims(b,axis=0)

            if return_loss == True:
                W = np.sum(Gamma*C) 
                loss.append(W)
                
        if return_loss == True:
            if return_map == True:
                return Gamma, loss
            else:
                return loss
        else:
            if return_map == True:
                return Gamma
            else:
                return None
