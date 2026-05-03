import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import numpy as np

from components.language_processing.language_processing import LanguageProcessing

from dotenv import load_dotenv
load_dotenv()

roberta_base_model: dict = {"vi": "vinai/phobert-base-v2", "en": "bert-base-uncased"}

class BERTWeighted(nn.Module):
    def __init__(self, pretrained_model_name='bert-base_uncased'):
        super(BERTWeighted, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        word_weights = self.linear(embeddings).squeeze(-1)
        word_weights = word_weights.masked_fill(attention_mask == 0, float('-inf'))
        batch_size, seq_len = attention_mask.shape
        for i in range(batch_size):
            mask_indices = attention_mask[i].nonzero(as_tuple=True)[0]
            first_pos = mask_indices[0]
            last_pos = mask_indices[-1]
            word_weights[i, first_pos] = float('-inf')
            word_weights[i, last_pos] = float('-inf')
        word_weights = torch.nn.functional.softmax(word_weights, dim=-1)
        return word_weights

def get_language_processor(language: str) -> LanguageProcessing:
    from components.language_processing.impl.english_language_processing import EnglishLanguageProcessing
    from components.language_processing.impl.khmer_language_processing import KhmerLanguageProcessing
    from components.language_processing.impl.vietnamese_language_processing import VietnameseLanguageProcessing

    languages = {
        "en": EnglishLanguageProcessing,
        "vi": VietnameseLanguageProcessing,
        "km": KhmerLanguageProcessing
    }

    language_class = languages.get(language.lower())
    
    if language_class:
        return language_class()
    else:
        raise ValueError(f"Unsupported language: {language}")


def pos_neg_samples_gen_first_round(
    document_id_answer: str,
    id_relevant_score_pairs: list[tuple[str, float]],
    negative_samples_limit: int = 35
):
    negative_samples_limit = max(0, negative_samples_limit)
    samples: list[tuple[str, float]] = []
    negative_samples_count: int = 0
    for id, relevant_score in id_relevant_score_pairs:
        if id == document_id_answer:
            samples.append((id, relevant_score))
        elif negative_samples_count < negative_samples_limit:
            samples.append((id, relevant_score))
            negative_samples_count += 1
    return samples


def pos_neg_samples_gen_later_round(
    previous_round_output: Tensor,
    previous_round_label_list: list[float],
    previous_round_similarity_score_list: list[float],
    previous_round_doc_chunk_list: list[list[str]],
    negative_samples_limit: int = 0
):
    # Separate streams by label
    pos_label_indices: list[int] = [i for i, label in enumerate(
        previous_round_label_list) if label == 1.0]
    neg_label_indices: list[int] = [i for i, label in enumerate(
        previous_round_label_list) if label == 0.0]

    # Get all label-1 streams and their indices
    pos_doc_chunk_list: list[list[str]] = [
        previous_round_doc_chunk_list[i] for i in pos_label_indices]
    pos_label_list: list[float] = [previous_round_label_list[i]
                                   for i in pos_label_indices]
    pos_similarity_score_list: list[float] = [
        previous_round_similarity_score_list[i] for i in pos_label_indices]

    # Get top-k label-0 streams
    neg_output: Tensor = previous_round_output[neg_label_indices]
    topk_values, topk_indices = torch.topk(neg_output, min(negative_samples_limit, len(neg_label_indices)))

    # Map topk_indices back to original indices
    topk_original_indices = [neg_label_indices[i] for i in topk_indices]

    # Get top-k streams and labels for label-0
    neg_doc_chunk_list = [previous_round_doc_chunk_list[i]
                          for i in topk_original_indices]
    neg_label_list = [previous_round_label_list[i]
                      for i in topk_original_indices]
    neg_similarity_score_list: list[float] = [
        previous_round_similarity_score_list[i] for i in topk_original_indices]

    # Combine label-1 and top-k label-0 streams
    doc_chunk_list: list[list[str]] = pos_doc_chunk_list + neg_doc_chunk_list
    label_list: list[float] = pos_label_list + neg_label_list
    similarity_score_list = pos_similarity_score_list + neg_similarity_score_list

    return doc_chunk_list, label_list, similarity_score_list

def min_max_scale(
    doc_list: list[tuple[str, float]], 
    a: float = 0.1, 
    b: float = 1.0
) -> list[tuple[str, float]]:
    scores = [score for _, score in doc_list]
    min_score = min(scores)
    max_score = max(scores)
    scaled_doc_list = [
        (id, a + (score - min_score) * (b - a) / (max_score - min_score) if max_score != min_score else a)
        for id, score in doc_list
    ]
    return scaled_doc_list


def combine_doc_list(
    doc_list_original_query: list[tuple[str, float]],
    doc_list_extended_query: list[tuple[str, float]]
):
    scaled_original_query = min_max_scale(doc_list_original_query)
    scaled_extended_query = min_max_scale(doc_list_extended_query)

    combined_doc_dict: dict[str, float] = {}
    for id, relevant_score in scaled_original_query:
        combined_doc_dict[id] = relevant_score
    for id, relevant_score in scaled_extended_query:
        if id not in combined_doc_dict or combined_doc_dict[id] < relevant_score:
            combined_doc_dict[id] = relevant_score
    combined_doc_list: list[tuple[str, float]] = [
        (id, relevant_score) for id, relevant_score in combined_doc_dict.items()]
    return combined_doc_list


def term_frequency(term, doc):
    return doc.count(term)


def inverse_doc_frequency(term, doc_list):
    N = len(doc_list)
    df = sum([1 for doc in doc_list if term in doc])
    return np.log((N - df + 0.5) / (df + 0.5) + 1)


def bm25(query, doc, doc_list, avgdl, k1=1.5, b=0.75, delta=1.0):
    score = 0
    doc_len = len(doc)

    for term in query:
        tf = term_frequency(term, doc)
        idf = inverse_doc_frequency(term, doc_list)

        numerator = (tf + delta) * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))

        score += idf * (numerator / denominator)

    return score

def compute_cosine_cost_matrix(source_embeddings: Tensor, target_embeddings: Tensor) -> Tensor:
    cosine_sim = torch.matmul(source_embeddings, target_embeddings.T)
    cosine_sim = F.normalize(cosine_sim, p=2, dim=-1)
    cost_matrix = 1 - cosine_sim 

    return cost_matrix

def pad_sentences(source: list[str], target: list[str], source_pad_token: str, target_pad_token: str) -> tuple[list[str], list[str], dict]:
    """
    Pad the shorter sentence with mask tokens to match the length of the longer sentence.
    
    Args:
        source (list[str]): List of tokens for source sentence
        target (list[str]): List of tokens for target sentence
        pad_token (str): The token used for padding
        
    Returns:
        Tuple of (padded source tokens, padded target tokens, attention mask)
    """
    max_len = max(len(source), len(target))
    
    padded_source = source + [source_pad_token] * (max_len - len(source))
    padded_target = target + [target_pad_token] * (max_len - len(target))
    
    return padded_source, padded_target

def tf_idf(term, doc, doc_list):
    return term_frequency(term, doc) * inverse_doc_frequency(term, doc_list)

def l1_normalize(tensor: Tensor):
    sum_val = tensor.sum().to(tensor.device)
    normalized = tensor / sum_val if sum_val != 0 else tensor
    return normalized.to(tensor.device).requires_grad_(tensor.requires_grad)

def tf_idf_dist(tokens, doc, doc_list, device='cpu'):
    return torch.tensor([tf_idf(token, doc, doc_list) for token in tokens], requires_grad=True, device=device, dtype=torch.float32)

def uniform_dist(sentence, device='cpu'):
    return torch.ones(len(sentence), requires_grad=True, device=device)

def load_roberta(language, device='cpu'):
    """Load tokenizer and model once and return them."""
    base_model = roberta_base_model[language]
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=os.getenv("HUGGINGFACE_TOKEN"))

    model = BERTWeighted(base_model)
    model.load_state_dict(torch.load(os.getenv("PROJECT_DIR") + "roberta_weighted/" + language + "_roberta_token_weight.pth",
                                     map_location=torch.device(device)))
    model.to(device)
    model.eval()
    
    return tokenizer, model

def roberta_dist(tokens, tokenizer, model, device='cpu'):      
    inputs = tokenizer(tokens, truncation=True, padding='max_length', max_length=128, return_tensors="pt", is_split_into_words=True)
    inputs = inputs.to(device)

    with torch.no_grad():
        word_weights = model(inputs["input_ids"], inputs["attention_mask"])

        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        predicted_sum = 0
        result = []
        for token, weight in zip(tokens, word_weights[0].cpu().numpy()):
            if token not in tokenizer.all_special_tokens:
                # print(f"Token: {token}, Prediction: {weight}")
                result.append(weight)
                predicted_sum += weight
        # print(f"Sum of Predicted Weights: {predicted_sum}")
        # print("\n" + "-"*50 + "\n")
        if (predicted_sum != 1):
            diff = 1 - predicted_sum
    return torch.tensor([float(weight + diff/len(result)) for weight in result], requires_grad=True, device=device)

def is_relevant(query, document):
    pass

def average_precision(query, retrieved_docs):
    num_relevant = 0
    precision_sum = 0.0

    for rank, doc in enumerate(retrieved_docs, start=1):
        if is_relevant(query, doc):
            num_relevant += 1
            precision = num_relevant / rank
            precision_sum += precision

    if num_relevant == 0:
        return 0.0

    return precision_sum / num_relevant

def mean_average_precision(queries, retrieved_docs_by_queries):
    sum_average_precision = 0.0
    for i, query in enumerate(queries):
        sum_average_precision += average_precision(query, retrieved_docs_by_queries[i])
    
    return sum_average_precision / len(queries)

def precision_at_k(query, retrieved_docs, k=10):
    top_k = retrieved_docs[:k]
    relevant_count = sum(1 for doc in top_k if is_relevant(query, doc))
    return relevant_count / k

def mean_precision_at_k(queries, retrieved_docs_by_queries, k=10):
    sum_precision_scores = 0
    for i, query in enumerate(queries):
        sum_precision_scores += precision_at_k(query, retrieved_docs_by_queries[i], k)

    return sum_precision_scores / len(queries) 

def recall_at_k(query, retrieved_docs, known_positive, k=10):
    top_k = retrieved_docs[:k]
    return 1.0 if known_positive in top_k else 0.0

def reciprocal_rank(query, retrieved_docs, known_positive):
    for idx, doc in enumerate(retrieved_docs, start=1):
        if doc == known_positive:
            return 1.0 / idx
    return 0.0

def mean_reciprocal_rank(queries, retrieved_docs_by_queries, known_positives):
    sum_reciprocal_rank = 0
    for i, query in enumerate(queries):
        sum_reciprocal_rank += reciprocal_rank(query, retrieved_docs_by_queries[i], known_positives[i])
    
    return sum_reciprocal_rank / len(queries)
