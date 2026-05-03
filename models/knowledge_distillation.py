from pandas import read_csv
import torch
from torch import Tensor
from sentence_transformers import SentenceTransformer
from components.ot_solver import OTSolver
from utils.utils import compute_cosine_cost_matrix, load_roberta, pad_sentences, tf_idf_dist, uniform_dist, l1_normalize, roberta_dist
from dotenv import load_dotenv
import os

load_dotenv()

class KnowledgeDistillation:
    def __init__(
        self, 
        teacher_model_language: str,
        student_model_language: str,
        teacher_model: str = "distiluse-base-multilingual-cased-v2",
        student_model: str = "distiluse-base-multilingual-cased-v2",
        bitext_data: str = os.getenv("PROJECT_DIR") + "bitext.csv",
        save_dir: str = os.getenv("PROJECT_DIR"),
        distribution: str = "padded_uniform", 
        device: str = "gpu",
        batch_size: int = 32,
        epochs: int = 4,
        learning_rate: float = 1e-5,
        epsilon: float = 0.1,
        log_every: int = 1,
        print_plan: bool = False,
    ) -> None:
        """
        Args:
            teacher_model_language (str): The language used in the monolingual training phase.
            student_model_language (str): The language to be learned by the student model.
            teacher_model (str): The base model for the teacher

            bitext_data (str): Path to the CSV file with parallel sentences.
            save_dir (str): The folder to save the trained model

            distribution (str): The distribution for tokens in the sentences

            device (str): Device (like "cuda", "cpu", "mps", "npu") that indicate where all models and computations run.

            batch_size (int): Determine how many sentence chunks should be encode at once in SentenceTransformer.

            epochs (int): Indicate total number of iterations of all the training data.
  
            learning_rate (float): The learning rate of the traning process.

            epsilon (float): Regularization parameter for optimal transport
        """
        self.teacher_model_language: str = teacher_model_language
        self.student_model_language: str = student_model_language

        self.bitext_data: str = bitext_data
        self.save_dir: str = save_dir

        self.distribution = distribution

        self.device: str = device
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate
        self.epsilon: float = epsilon
        self.log_every: int = log_every
        self.print_plan: bool = print_plan

        self.teacher: SentenceTransformer = SentenceTransformer(teacher_model, device=self.device, token=os.getenv("HUGGINGFACE_TOKEN"))
        self.student: SentenceTransformer = SentenceTransformer(student_model, device=self.device, token=os.getenv("HUGGINGFACE_TOKEN"))
        self.teacher.max_seq_length = 512
        self.student.max_seq_length = 512
        for param in self.teacher.parameters():
            param.requires_grad = False
    
        self.teacher.to(self.device)
        self.student.to(self.device)

        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.learning_rate)

        self.ot_solver: OTSolver = OTSolver(self.device)

    def optical(self, source_sentence: str, target_sentence: str):
        source_tokens = self.teacher.tokenizer.tokenize(source_sentence)
        target_tokens = self.student.tokenizer.tokenize(target_sentence)

        # Guard: skip degenerate sentences with no tokens
        if not source_tokens or not target_tokens:
            dummy = torch.tensor(0.0, device=self.device, requires_grad=True)
            return None, dummy

        self.optimizer.zero_grad()  # Reset gradients at the start of each sample

        if "padded" in self.distribution:
            source_tokens, target_tokens = pad_sentences(source_tokens, target_tokens, self.teacher.tokenizer.pad_token, self.student.tokenizer.pad_token)

        if self.distribution == "tf_idf":
            source_dist = tf_idf_dist(source_tokens, source_sentence, self.source_sentence_list, self.device)
            target_dist = tf_idf_dist(target_tokens, target_sentence, self.target_sentence_list, self.device)
        elif self.distribution == "roberta":
            source_dist = roberta_dist(source_tokens, self.teacher_tokenizer, self.teacher_roberta_model, self.device)
            target_dist = roberta_dist(target_tokens, self.student_tokenizer, self.student_roberta_model, self.device)
        elif "uniform" in self.distribution:
            source_dist = uniform_dist(source_tokens, self.device)
            target_dist = uniform_dist(target_tokens, self.device)

        source_dist = l1_normalize(source_dist)
        target_dist = l1_normalize(target_dist)  

        source_ids = self.teacher.tokenizer.convert_tokens_to_ids(source_tokens)
        source_ids = torch.tensor([source_ids], device=self.device)
        attention_mask = [1 if token != self.teacher.tokenizer.pad_token else 0 for token in source_tokens]
        attention_mask = torch.tensor([attention_mask], device=self.device)
        source_encoded = {
            'input_ids': source_ids,
            'attention_mask': attention_mask
        }

        target_ids = self.student.tokenizer.convert_tokens_to_ids(target_tokens)
        target_ids = torch.tensor([target_ids], device=self.device)
        attention_mask = [1 if token != self.student.tokenizer.pad_token else 0 for token in target_tokens]
        attention_mask = torch.tensor([attention_mask], device=self.device)
        target_encoded = {
            'input_ids': target_ids,
            'attention_mask': attention_mask
        }

        with torch.no_grad():
            source_embeddings: Tensor = self.teacher.forward(source_encoded)['token_embeddings'].squeeze(0)

        target_embeddings: Tensor = self.student.forward(target_encoded)['token_embeddings'].squeeze(0)

        # Guard: check token embeddings for NaN/Inf before building cost matrix
        if (torch.isnan(source_embeddings).any() or torch.isinf(source_embeddings).any() or
                torch.isnan(target_embeddings).any() or torch.isinf(target_embeddings).any()):
            dummy = torch.tensor(0.0, device=self.device, requires_grad=True)
            return None, dummy

        cost: Tensor = compute_cosine_cost_matrix(source_embeddings, target_embeddings)

        # Guard: cost matrix must be finite for Sinkhorn to converge
        if torch.isnan(cost).any() or torch.isinf(cost).any():
            dummy = torch.tensor(0.0, device=self.device, requires_grad=True)
            return None, dummy

        # Guard: distributions must be valid (no NaN, sum > 0)
        if (torch.isnan(source_dist).any() or torch.isnan(target_dist).any() or
                source_dist.sum() < 1e-8 or target_dist.sum() < 1e-8):
            dummy = torch.tensor(0.0, device=self.device, requires_grad=True)
            return None, dummy

        plan, loss = self.ot_solver(source_dist, target_dist, cost)

        return plan, loss

    def train_loop(self, source_sentence: str, target_sentence: str):
        plan, loss = self.optical(source_sentence, target_sentence)

        # Guard: optical() returns a dummy zero tensor when inputs are degenerate
        if plan is None or torch.isnan(loss) or torch.isinf(loss):
            return plan, loss
        loss.backward()
        # Clip gradients to prevent exploding-gradient NaN cascade in weights
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()
        return plan, loss

    def train(self) -> str:
        """
        Train the student model using knowledge distillation.

        Returns:
            str: The path to the multilingual sentence transformer
        """
        print("Start training")
        for epoch in range(self.epochs):
            df = read_csv(self.bitext_data)
            bitext_data = list(zip(df["source"], df["target"]))
            if self.distribution == "roberta":
                self.teacher_tokenizer, self.teacher_roberta_model = load_roberta(self.teacher_model_language, self.device)
                self.student_tokenizer, self.student_roberta_model = load_roberta(self.student_model_language, self.device)
            elif self.distribution == "tf_idf":
                self.source_sentence_list = [source_sentence for source_sentence, _ in bitext_data]
                self.target_sentence_list = [target_sentence for _, target_sentence in bitext_data]
            total_steps = len(bitext_data)
            for step, (source_sentence, target_sentence) in enumerate(bitext_data, start=1):
                plan, loss = self.train_loop(source_sentence, target_sentence)
                if self.log_every and step % self.log_every == 0:
                    if loss is None or torch.isnan(loss) or torch.isinf(loss):
                        print(f"epoch={epoch + 1}/{self.epochs} step={step}/{total_steps} loss=skipped")
                    else:
                        print(f"epoch={epoch + 1}/{self.epochs} step={step}/{total_steps} loss={loss.item():.6f}")
                        if self.print_plan and plan is not None:
                            print(plan.detach().cpu())


        self.student.save(self.save_dir + f"sentence_transformer_multilingual_" + self.distribution)
        check_point = {
            'student_sentence_transformer_save_path': self.save_dir + f"sentence_transformer_multilingual_" + self.distribution

        }
        torch.save(check_point, self.save_dir + f"sentence_transformer_multilingual_"  + self.distribution + '/model.pth')

        return self.save_dir + f"sentence_transformer_multilingual_" + self.distribution
