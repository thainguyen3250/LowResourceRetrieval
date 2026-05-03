import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity, relu
from typing import List
import concurrent.futures


class CustomSentenceTransformer(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            device: str = "cpu",
            batch_size: int = 256,
            is_multilingual_retrival: bool = False,
            pretrained_model_name_or_path_for_query: str = "",
            max_workers: int = 35,
            encoding_workers: int = 2
    ):
        super(CustomSentenceTransformer, self).__init__()

        _on_cuda = device.startswith("cuda")
        primary_device = "cuda:0" if _on_cuda else device

        self.document_sentence_transformer = SentenceTransformer(
            pretrained_model_name_or_path, device=primary_device)
        self.document_sentence_transformer.max_seq_length = 256

        if not is_multilingual_retrival:
            self.query_sentence_transformer = self.document_sentence_transformer
        else:
            self.query_sentence_transformer = SentenceTransformer(
                pretrained_model_name_or_path_for_query, device=primary_device)
            self.query_sentence_transformer.max_seq_length = 256

        self.device = primary_device
        # Force batch size to be at most 8 on CUDA to prevent OOM
        if _on_cuda and batch_size > 256:
            self.batch_size = 256
        else:
            self.batch_size = batch_size

        # Dual-GPU: load a second read-only encoder on cuda:1.
        # We use exactly 2 threads — one per GPU — each encoding streams
        # SEQUENTIALLY, so at most 1 forward pass is active per GPU at any time.
        if _on_cuda and torch.cuda.device_count() >= 2:
            self.device2 = "cuda:1"
            self.document_sentence_transformer2 = SentenceTransformer(
                pretrained_model_name_or_path, device=self.device2)
            self.document_sentence_transformer2.max_seq_length = 256
            print(f"[CustomSentenceTransformer] Dual-GPU: "
                  f"{torch.cuda.get_device_name(0)} / {torch.cuda.get_device_name(1)}")
        else:
            self.device2 = None
            self.document_sentence_transformer2 = None

        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=1, bias=True),
            nn.Sigmoid()
        ).to(device=self.device)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _process_group(
            self,
            query_embedding: Tensor,
            indexed_streams: List[tuple],
            model: SentenceTransformer,
            encode_device: str
    ) -> dict:
        """Process (index, stream) pairs efficiently using batched tensor operations.
        
        This flattens all chunks from all streams into a single massive list, 
        encodes them at once, and unpacks the results. This eliminates Python
        thread overhead and maxes out GPU utilization.
        """
        results = {}
        if not indexed_streams:
            return results

        # Flatten all sentences
        all_sentences = []
        stream_slices = {}  # idx -> (start, end)
        
        current_idx = 0
        for idx, stream in indexed_streams:
            if not stream:
                results[idx] = 0.0
                continue
            start = current_idx
            all_sentences.extend(stream)
            end = len(all_sentences)
            stream_slices[idx] = (start, end)
            current_idx = end
            
        if not all_sentences:
            return results

        q_emb = (query_embedding if str(query_embedding.device) == encode_device
                 else query_embedding.to(encode_device))

        try:
            with torch.no_grad():
                # Encode everything at once using SentenceTransformer's internal batching
                # A batch_size of 128 or 256 easily fits on 15 GB T4 and maxes compute.
                all_embeddings = model.encode(
                    all_sentences, 
                    batch_size=128, 
                    convert_to_tensor=True, 
                    device=encode_device,
                    show_progress_bar=False
                )
                
                # If embeddings are empty
                if all_embeddings.numel() == 0:
                    for idx in stream_slices.keys():
                        results[idx] = 0.0
                    return results
                
                # Compute similarities for all sentences at once
                sims = cosine_similarity(all_embeddings, q_emb.unsqueeze(0)).squeeze()
                if sims.dim() == 0:
                    sims = sims.unsqueeze(0)
                    
                sims = torch.nan_to_num(sims, nan=0.0)
                sims = relu(sims)
                
                # Split back into individual streams
                for idx, (start, end) in stream_slices.items():
                    stream_sims = sims[start:end]
                    # Compute probabilty union: 1 - prod(1 - stream_sims)
                    result = (1 - torch.prod(1 - stream_sims)).item()
                    results[idx] = result
                    
        except RuntimeError as e:
            print(f"Error processing group on {encode_device}: {e}")
            for idx in stream_slices.keys():
                results[idx] = 0.0
                
        return results

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(
            self,
            preprocessed_query: str,
            lexical_or_topic_similarities: List[float],
            sentence_streams: List[List[str]]
    ) -> Tensor:
        if isinstance(preprocessed_query, list):
            preprocessed_query = " ".join(preprocessed_query)
            
        with torch.no_grad():
            query_embedding = self.query_sentence_transformer.encode(
                preprocessed_query, convert_to_tensor=True, device=self.device)

        all_semantic_similarities = [0.0] * len(sentence_streams)

        if self.device2 is not None:
            # Split streams: even → GPU 0, odd → GPU 1
            # 2 threads run in parallel, each encodes its streams one-by-one
            group0 = [(i, s) for i, s in enumerate(sentence_streams) if i % 2 == 0]
            group1 = [(i, s) for i, s in enumerate(sentence_streams) if i % 2 == 1]

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                f0 = ex.submit(self._process_group, query_embedding, group0,
                               self.document_sentence_transformer, self.device)
                f1 = ex.submit(self._process_group, query_embedding, group1,
                               self.document_sentence_transformer2, self.device2)
                combined = {**f0.result(), **f1.result()}

            for idx, val in combined.items():
                all_semantic_similarities[idx] = val
        else:
            # Single GPU: fully sequential — safest for memory
            for i, stream in enumerate(sentence_streams):
                try:
                    all_semantic_similarities[i] = self._encode_stream(
                        query_embedding, stream,
                        self.document_sentence_transformer, self.device)
                except RuntimeError as e:
                    print(f"Error processing stream {i}: {e}")
                finally:
                    if self.device.startswith("cuda"):
                        torch.cuda.empty_cache()

        sem_tensor = torch.tensor(all_semantic_similarities, device=self.device)
        lex_tensor = torch.tensor(lexical_or_topic_similarities, device=self.device)
        combined_tensor = torch.stack((sem_tensor, lex_tensor), dim=1)
        return self.linear_sigmoid_stack(combined_tensor).squeeze()
