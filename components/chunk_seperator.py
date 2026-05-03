from functools import reduce

from .dataset import DocumentDataset


class ChunkSeperator:
    def __init__(
            self,
            document_dataset: DocumentDataset,
            chunk_length_limit: int = 128,
    ) -> None:
        """
        Args:
            document_dataset (str): An object of DocumentDataset class, represent a document dataset\
            that we want to get documents and seperate them into chunks
            chunk_length_limit (int): The limit length of each chunk                
        """
        self.document_dataset: DocumentDataset = document_dataset
        self.chunk_length_limit: int = max(1, chunk_length_limit)

    def get_chunks_of_document(self, id: str) -> list[str]:
        """
        Get the chunk list of the document.

        Args:
            id (str): Id of the document we want to and seperate them into chunks

        Returns:
            list[str]: Chunk list of the document. Each chunk consist one or more sentences that total number of tokens\
            in each chunk is less than the `chunk_length_limit`.
        """
        preprocessed_title, preprocessed_content, _, _, _, _, _ = self.document_dataset.get_item_by_id(id)
        document_chunks: list[str] = preprocessed_title
        current_chunk: str = ""
        current_chunk_token_count: int = 0
        
        for preprocessed_sentence in preprocessed_content:
            tokenized_sentence = self.document_dataset.language_processing.tokenizer(preprocessed_sentence)
            if current_chunk_token_count + len(tokenized_sentence) > self.chunk_length_limit:
                document_chunks.append(current_chunk)
                current_chunk = preprocessed_sentence
                current_chunk_token_count = len(tokenized_sentence)
            else:
                current_chunk += preprocessed_sentence
                current_chunk_token_count += len(tokenized_sentence)
        
        return document_chunks
            
            
        

        
        

