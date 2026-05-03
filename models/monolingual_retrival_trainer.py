from functools import reduce
import os
import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from components import LanguageProcessing
from components import DocumentDataset
from components import QueryDocDataset
from components import QueryExpansion
from components import LexicalMatching
from components import ChunkSeperator
from components import CustomSentenceTransformer
from components import FineTuneLanguageModel
from utils.utils import pos_neg_samples_gen_first_round, pos_neg_samples_gen_later_round, combine_doc_list, get_language_processor

# These control how many hard-negative documents are evaluated per round.
# Round 1: up to 35 negatives from BM25 top results
# Round 2: top 20 hard-negatives from Round 1 output
# Round 3: top 15 hard-negatives from Round 2 output
FIRST_ROUND_NEGATIVE_SAMPLE_COUNT  = 35
SECOND_ROUND_NEGATIVE_SAMPLE_COUNT = 20
THIRD_ROUND_NEGATIVE_SAMPLE_COUNT  = 15

def custom_collate_fn(batch):
    queries = [item[0] for item in batch]  # List of preprocessed queries
    documents = [item[1] for item in batch]  # List of tokenized queries
    scores = [item[2] for item in batch]  # List of document id
    
    return queries, documents, scores

class ContrastiveLoss(nn.Module):
    """
    This class represents the contrastive loss function and is used to calcualate the loss value.
    """
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs: Tensor, labels: Tensor) -> Tensor:
        """
        outputs: Tensor of shape (N,)
        labels: Tensor of shape (N,), where 1 indicates relevant and 0 indicates non-relevant
        """
        # Compute the positive pair loss (y * d^2)
        positive_loss: Tensor = labels * outputs ** 2

        # Compute the negative pair loss ((1 - y) * max(0, margin - d)^2)
        margin_diff: Tensor = torch.relu(self.margin - outputs)
        negative_loss: Tensor = (1 - labels) * margin_diff ** 2

        # Combine positive and negative losses
        loss: Tensor = 0.5 * (positive_loss + negative_loss).mean()
        return loss


class MonolingualRetrivalTrainer:
    def __init__(
        self,
        document_dir: str,
        processed_doc_store_dir: str,
        qd_dir: str,
        pretrained_model_name_or_path: str,
        checkpoint_path: str = None,
        do_mlm_fine_tune: bool = False,
        language: str = 'vi',
        chunk_length_limit: int = 128,
        device: str = "cuda",
        batch_size: int = 256,
        margin: float = 1.0,
        learning_rate: float = 1e-5,
        epochs: int = 4
    ) -> None:
        """
        Args:
            document_dir (str): ABSOLUTE path to the directory containing documents with title, topic, and content xml-tag.

            processed_doc_store_dir (str): processed_doc_store_dir (str): ABSOLUTE path to the directory where you want to store preprocessed-documents.

            qd_dir (str): Path to the folder containing CSV files with queries and corresponding answer document file paths (ABSOLUTE path).

            pretrained_model_name_or_path (str): A string - the model id of a pretrained language model hosted\
            inside a model repo on huggingface.co (e.g: `vinai/phobert-base-v2`, `FacebookAI/roberta-base`,...).\
            OR a path to a directory containing your own language model. This model should be based on a transformer model\
            such as BERT, RoBERTa, or other Hugging Face models.

            do_mlm_fine_tune (bool): Indicator variable telling whether or not to do mlm fine-tune task for the language model.

            language (str): Language of the query and documents. Since this is class for monolingual-training, the language\
            of the query and documents must be the same.

            language_processing (LanguageProcessing): Language processing object for the language.

            chunk_length_limit (int): The limit length of each chunk. Representing the max number of tokens in each chunk\
            when seperating the document.

            device (str): Device (like "cuda", "cpu", "mps", "npu") that indicate where all models and computations run.

            batch_size (int): Determine how many sentence chunks should be encode at once in SentenceTransformer.

            margin (float): The value of the margin in the contrastive-loss equation.

            learning_rate (float): The learning rate of the traning process.

            epochs (int): Indicate total number of iterations of all the training data.
        """
        super().__init__()
        self.language = language
        language_processing = get_language_processor(language)
        self.document_dataset: DocumentDataset = DocumentDataset(
            document_dir,
            processed_doc_store_dir,
            language,
        )
        self.query_doc_dataset = QueryDocDataset(
            qd_dir,
            language,
        )
        self.query_expansion: QueryExpansion = QueryExpansion(
            self.document_dataset)
        self.lexical_matching: LexicalMatching = LexicalMatching(self.document_dataset)
        self.chunk_seperator: ChunkSeperator = ChunkSeperator(
            self.document_dataset,
            chunk_length_limit
        )
        self.base_language_model: str = pretrained_model_name_or_path
        if do_mlm_fine_tune and checkpoint_path is None:
            fine_tune_language_model = FineTuneLanguageModel(
                self.document_dataset, pretrained_model_name_or_path)
            self.base_language_model = fine_tune_language_model.train()
        # Force batch_size to be at most 256 on CUDA to prevent OOM
        if device.startswith("cuda") and batch_size > 256:
            print(f"[MonolingualRetrivalTrainer] Capping batch_size from {batch_size} to 256 for CUDA.")
            batch_size = 256
            
        if checkpoint_path is not None:
            print(f"[MonolingualRetrivalTrainer] Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
            self.custom_sentence_transformer = CustomSentenceTransformer(
                checkpoint['sentence_transformer_save_path'],
                device,
                batch_size
            ).to(device=device)
            self.custom_sentence_transformer.linear_sigmoid_stack.load_state_dict(
                checkpoint['linear_sigmoid_stack'])
        else:
            self.custom_sentence_transformer = CustomSentenceTransformer(
                self.base_language_model,
                device,
                batch_size
            ).to(device=device)

        self.contrastive_loss_fn: ContrastiveLoss = ContrastiveLoss(margin).to(device)
        self.optimizer = Adam(
            self.custom_sentence_transformer.parameters(), lr=learning_rate)
        self.epochs: int = epochs
        self.device: str = device
        self.batch_size: int = batch_size
        self.language_processing: LanguageProcessing = language_processing
        # How often to save a mid-epoch checkpoint (every N batches).
        # Set to 0 to disable. Useful on Kaggle where sessions can crash.
        self.checkpoint_every_n_batches: int = 50
            

    def train(self) -> tuple[str, str]:
        """
        Training entry point.

        Returns:
            tuple[str,str]: The first string is the path to the fine-tuned `SentenceTransformer` model.\
            This can be loaded separately to do Knowledge Distillation later.\
            The second string is the path to the whole `CustomSentenceTransformer` model. Because `CustomSentenceTransformer`\
            has an attribute of class `SentenceTransformer`, which have different way to load the model,\
            we save it as a dictionary of two keys represents two part of `CustomSentenceTransformer`.
        """
        # Use the configured batch_size (was hardcoded to 32, ignoring self.batch_size)
        query_doc_dataloader = DataLoader(
            self.query_doc_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn
        )
        sentence_transformer_save_path: str = (
            os.getenv("PROJECT_DIR") + f"sentence_transformer_finetune/{self.language}")
        custom_sentence_transformer_save_path: str = (
            os.getenv("PROJECT_DIR") + f"custom_sentence_transformer_trained/{self.language}")

        print("Training started...")
        for epoch in range(self.epochs):
            print("-----------------------------")
            print(f"Epoch {epoch + 1}/{self.epochs}:")
            print("-----------------------------")
            for batch_count, sample_batch in enumerate(query_doc_dataloader):
                print(f"Batch {batch_count + 1} started.")
                loss = self._run_training_sample_batch(sample_batch)
                print(f"Batch {batch_count + 1} completed. Loss: {loss.item():.6f}")

                # ── Mid-epoch checkpoint ───────────────────────────────────────
                # Saves progress periodically so a GPU OOM later in the epoch
                # does not lose all trained weights.
                if (self.checkpoint_every_n_batches > 0
                        and (batch_count + 1) % self.checkpoint_every_n_batches == 0):
                    self.custom_sentence_transformer.document_sentence_transformer.save(
                        sentence_transformer_save_path)
                    ckpt = {
                        'sentence_transformer_save_path': sentence_transformer_save_path,
                        'linear_sigmoid_stack':
                            self.custom_sentence_transformer.linear_sigmoid_stack.state_dict()
                    }
                    torch.save(ckpt, custom_sentence_transformer_save_path + '/model.pth')
                    print(f"  [Checkpoint] saved at epoch {epoch+1}, batch {batch_count+1}")

        # ── Final save ────────────────────────────────────────────────────────
        self.custom_sentence_transformer.document_sentence_transformer.save(
            sentence_transformer_save_path)
        check_point = {
            'sentence_transformer_save_path': sentence_transformer_save_path,
            'linear_sigmoid_stack':
                self.custom_sentence_transformer.linear_sigmoid_stack.state_dict()
        }
        torch.save(check_point, custom_sentence_transformer_save_path + '/model.pth')

        print("Training completed.")
        print(f"Fine-tuned SentenceTransformer model saved at: {sentence_transformer_save_path}")
        print(f"CustomSentenceTransformer model saved at: {custom_sentence_transformer_save_path}")

        return sentence_transformer_save_path, custom_sentence_transformer_save_path
    
    def _run_training_sample_batch(
        self, 
        sample_batch: tuple[list[list[str]], list[list[str]], list[str]]
    ) -> Tensor:
        """
        Runs a single training batch for the monolingual retrieval model.
        Args:
            sample_batch (tuple[list[list[str]], list[list[str]], list[str]]): \
                A tuple containing the list of preprocessed query, list of tokenized query, and list of document id.
        Returns:
            final_loss (Tensor): The final loss value for the batch.
        """
        sample_count: int = len(sample_batch[0])
        final_loss: Tensor = torch.tensor(0.0, device=self.device)
        for i in range(sample_count):
            print(f"Processing sample {i} in batch...")
            query_preprossed: list[str] = sample_batch[0][i]
            tokenized_query: list[str] = sample_batch[1][i]
            document_id: str = sample_batch[2][i]
            query_segmented: str = ""
            for part in query_preprossed:
              query_segmented += part
            print(f"Sample {i} in batch: data loaded...")
            extended_query: list[str] = tokenized_query + self.query_expansion.get_expansion_term(tokenized_query)
            original_query_doc_ranking: list[tuple[str, float]] = self.lexical_matching.get_documents_ranking(tokenized_query)
            print(f"Sample {i} in batch: lexical matching completed...")
            extended_query_doc_ranking: list[tuple[str, float]] = self.lexical_matching.get_documents_ranking(extended_query)
            original_query_relevant_doc_list: list[tuple[str, float]] = pos_neg_samples_gen_first_round(
                document_id, original_query_doc_ranking, FIRST_ROUND_NEGATIVE_SAMPLE_COUNT)
            print(f"Sample {i} in batch: positive/negative pair generated...")
            extended_query_relevant_doc_list: list[tuple[str, float]] = pos_neg_samples_gen_first_round(
                document_id, extended_query_doc_ranking, FIRST_ROUND_NEGATIVE_SAMPLE_COUNT)
            combine_lexical_relevant_doc_list: list[tuple[str, float]] = combine_doc_list(
                original_query_relevant_doc_list, extended_query_relevant_doc_list)

            lexical_relevant_doc_chunk_list: list[list[str]] = [self.chunk_seperator.get_chunks_of_document(pair[0]) 
                                                                for pair in combine_lexical_relevant_doc_list]
            print(f"Sample {i} in batch: chunk list created...")                                                  
            lexical_similarity_score_list: list[float] = [pair[1] for pair in combine_lexical_relevant_doc_list]

            self.custom_sentence_transformer.train()
            print(f"Sample {i} in batch: Sentence Trandformer started #1...")

            first_round_label_list: list[float] = [(1.0 if pair[0] == document_id else 0.0) for pair in combine_lexical_relevant_doc_list]
            print(f"Sample {i} in batch: Custom Sentence Trandformer round #1...")
            first_round_output, _ = self._run_training_custom_sentence_transformer_round(
                query_segmented,
                first_round_label_list,
                lexical_relevant_doc_chunk_list,
                lexical_similarity_score_list
            )
            # # Free cached GPU memory before round 2
            # if self.device.startswith("cuda"):
            #     torch.cuda.empty_cache()

            print(f"Sample {i} in batch: Sentence Trandformer started #2...")
            (second_round_doc_chunk_list,
             second_round_label_list,
             second_round_lexical_similarity_score_list) = pos_neg_samples_gen_later_round(
                 first_round_output,
                 first_round_label_list,
                 lexical_similarity_score_list,
                 lexical_relevant_doc_chunk_list,
                 SECOND_ROUND_NEGATIVE_SAMPLE_COUNT)
            # Guard: skip round 2 if no candidates remain
            if not second_round_doc_chunk_list:
                print(f"Sample {i}: no round-2 candidates, skipping rounds 2 & 3.")
                continue
            print(f"Sample {i} in batch: Custom Sentence Trandformer round #2...")
            second_round_output, _ = self._run_training_custom_sentence_transformer_round(
                query_segmented,
                second_round_label_list,
                second_round_doc_chunk_list,
                second_round_lexical_similarity_score_list)
            # Free cached GPU memory before round 3
            # if self.device.startswith("cuda"):
            #     torch.cuda.empty_cache()

            print(f"Sample {i} in batch: Sentence Trandformer started #3...")
            (third_round_doc_chunk_list,
             third_round_label_list,
             third_round_lexical_similarity_score_list) = pos_neg_samples_gen_later_round(
                    second_round_output, second_round_label_list,
                    second_round_lexical_similarity_score_list,
                    second_round_doc_chunk_list,
                    THIRD_ROUND_NEGATIVE_SAMPLE_COUNT)
            # Guard: skip round 3 if no candidates remain
            if not third_round_doc_chunk_list:
                print(f"Sample {i}: no round-3 candidates, skipping round 3.")
                continue
            print(f"Sample {i} in batch: Custom Sentence Trandformer round #3...")
            _, third_round_loss = self._run_training_custom_sentence_transformer_round(
                query_segmented,
                third_round_label_list,
                third_round_doc_chunk_list,
                third_round_lexical_similarity_score_list)
            # if self.device.startswith("cuda"):
            #     torch.cuda.empty_cache()

            final_loss = third_round_loss

        return final_loss
    
    def _run_training_custom_sentence_transformer_round(
        self,
        query_segmented: str,
        label_list: list[float],
        doc_chunk_list: list[list[str]],
        similarity_score_list: list[float]
    ) -> tuple[Tensor, Tensor]:
        """
        Runs a single training round for the sentence-transformer monolingual retrieval model.
        Args:
            query_segmented (str): The segmented query string.
            label_list (list[float]): A list of float labels corresponding to the relevance of each document chunk.
            doc_chunk_list (list[list[str]]): A list of document chunks, where each chunk is a list of strings.
            similarity_score_list (list[float]): A list of similarity scores for each document chunk.
        Returns:
            round_output (Tensor): The output tensor from the custom sentence transformer.
        """
        self.optimizer.zero_grad()
        label_tensor: Tensor = torch.tensor(label_list, device=self.device)

        # NOTE: fp16 autocast is intentionally disabled.
        # The sigmoid output ∈ (0,1) underflows to 0 in fp16, collapsing
        # cosine similarities and producing NaN gradients. Run in fp32.
        round_output: Tensor = self.custom_sentence_transformer(
            query_segmented,
            similarity_score_list,
            doc_chunk_list
        )

        # Guard against NaN/Inf in model output before computing loss
        if torch.isnan(round_output).any() or torch.isinf(round_output).any():
            print("  [Warning] NaN/Inf in round_output — skipping this round.")
            return round_output.detach(), torch.tensor(0.0, device=self.device)

        round_loss: Tensor = self.contrastive_loss_fn(round_output, label_tensor)

        if torch.isnan(round_loss) or torch.isinf(round_loss):
            print(f"  [Warning] NaN/Inf loss detected — skipping backward.")
            return round_output.detach(), torch.tensor(0.0, device=self.device)

        round_loss.backward()
        # Clip gradients to prevent exploding gradient NaN cascade
        torch.nn.utils.clip_grad_norm_(
            self.custom_sentence_transformer.parameters(), max_norm=1.0)
        self.optimizer.step()

        return round_output.detach(), round_loss
