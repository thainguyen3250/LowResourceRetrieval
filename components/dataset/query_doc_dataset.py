from functools import reduce
import os
from torch.utils.data import Dataset
import pandas as pd
from ..language_processing.language_processing import LanguageProcessing
from utils.utils import get_language_processor

class QueryDocDataset(Dataset):
    def __init__(
            self,
            qd_dir: str,
            language: str = 'vi',
    ):
        self.language: str = language
        self.language_processing: LanguageProcessing = get_language_processor(language)
        self.qd_dir: str = qd_dir
        self.qd_pairs = self._load_qd_pairs()

    def _load_qd_pairs(self):
        """
        Load query-document pairs from parquet files in the specified folder.
        Then, preprocess and tokenize the query of each pair.
        Returns:
            list: A list of tuples containing query-document pairs.
        """
        qd_pairs: list[tuple[list[str], list[str], str]] = []
        all_files = [os.path.join(self.qd_dir, f) for f in os.listdir(self.qd_dir) if f.endswith('.parquet')]
        
        for file in all_files:
            df = pd.read_parquet(file, engine='fastparquet')
            for idx, row in df.iterrows():
                query = row['query']
                document_id = row['id']
                preprocessed_query: list[str] = self.language_processing.text_preprocessing(query)
                query_tokenized: list[str] = reduce(lambda prev, curr: prev + self.language_processing.tokenizer(curr), preprocessed_query, [])
                
                qd_pairs.append((preprocessed_query, query_tokenized, document_id))
        
        return qd_pairs

    def __len__(self) -> int:
        return len(self.qd_pairs)

    def __getitem__(self, idx: int) -> tuple[list[str], list[str], str]:
        if idx >= len(self.qd_pairs):
            raise StopIteration
        preprocessed_query, query_tokenized, document_id = self.qd_pairs[idx]
        return preprocessed_query, query_tokenized, document_id