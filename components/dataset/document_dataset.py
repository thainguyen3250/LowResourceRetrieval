from functools import reduce
import os
import json
import pandas as pd
from torch.utils.data import Dataset
from ..language_processing.language_processing import LanguageProcessing
from utils.utils import get_language_processor

class DocumentDataset(Dataset):
    def __init__(
            self,
            parquet_file_dir: str,
            processed_doc_store_dir: str,
            language: str = 'vi'
    ):
        self.language: str = language
        self.language_processing: LanguageProcessing = get_language_processor(language)
        self.parquet_file_dir: str = parquet_file_dir
        self.processed_doc_store_dir: str = processed_doc_store_dir
        self.document_count: int = self._load_documents()

    def _load_documents(self) -> int:
        """
        Load documents from parquet files in the specified folder.
        Then, preprocess and tokenize the title and content of each document.
        Store the preprocessed and tokenized documents in JSON files.
        Returns:
            int: The number of documents loaded.
        """
        all_files = [os.path.join(self.parquet_file_dir, f) for f in os.listdir(self.parquet_file_dir) if f.endswith('.parquet')]
        document_count = 0
        
        for file in all_files:
            df = pd.read_parquet(file, engine='fastparquet')
            for idx, row in df.iterrows():
                title = row['title']
                content = row['text']
                document_id = row['id']
                url = row['url']
                title_preprocessed: list[str] = self.language_processing.text_preprocessing(title)
                content_preprocessed: list[str] = self.language_processing.text_preprocessing(content)
                title_tokenized: list[str] = reduce(lambda prev, curr: prev + self.language_processing.tokenizer(curr), title_preprocessed, [])
                content_tokenized: list[str] = reduce(lambda prev, curr: prev + self.language_processing.tokenizer(curr), content_preprocessed, [])
                document = {
                    "title": title_preprocessed,
                    "content": content_preprocessed,
                    "title_tokenized": title_tokenized,
                    "content_tokenized": content_tokenized,
                    "id": document_id,
                    "url": url,
                    "file_path": file
                }
                json_file_path = os.path.join(
                    self.processed_doc_store_dir, f"{document_count}.json")
                with open(json_file_path, 'w') as json_file:
                    json.dump(document, json_file)
                document_count += 1
    
        return document_count

    def __len__(self) -> int:
        return self.document_count

    def __getitem__(self, idx: int) -> tuple[list[str], list[str], list[str], list[str], str, str, str]:
        if idx >= self.document_count:
            raise StopIteration
        json_file_path = os.path.join(
            self.processed_doc_store_dir, f"{idx}.json")
        with open(json_file_path, 'r') as json_file:
            document = json.load(json_file)

        return (
            document["title"],
            document["content"],
            document["title_tokenized"],
            document["content_tokenized"],
            document["id"],
            document["url"],
            document["file_path"]
        )

    def get_item_by_id(self, document_id: str) -> tuple[list[str], list[str], list[str], list[str], str, str, str]:
        """
        Get the document by its id.
        Args:
            document_id (str): Id of the document we want to get.
        Returns:
            tuple: A tuple containing the title, content, tokenized title, tokenized content, id, url, and file path of the document.
        """
        for idx in range(self.document_count):
            json_file_path = os.path.join(
                self.processed_doc_store_dir, f"{idx}.json")
            with open(json_file_path, 'r') as json_file:
                document = json.load(json_file)
                if document["id"] == document_id:
                    return (
                        document["title"],
                        document["content"],
                        document["title_tokenized"],
                        document["content_tokenized"],
                        document["id"],
                        document["url"],
                        document["file_path"]
                    )
        raise ValueError(f"Document with id {document_id} not found")