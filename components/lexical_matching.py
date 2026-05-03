from functools import reduce
from numpy import argsort, float64
from rank_bm25 import BM25Plus

from .dataset import DocumentDataset


class LexicalMatching:
    def __init__(self, document_dataset: DocumentDataset) -> None:
        """
        Args:
            document_dataset (DocumentDataset): An object of DocumentDataset class, represent a document dataset that query wants to retrive.
        """
        self.document_dataset: DocumentDataset = document_dataset
        self.bm25plus, self.id_lst = self._load_bm25plus()
    
    def _load_bm25plus(self) -> tuple[BM25Plus, list[str]]:
        """
        Load BM25Plus instance initialized with documents from the document dataset, and list of id of each documents.

        Returns:
            tuple[BM25Plus,list[str]]: BM25Plus instance initialized with documents from the document dataset, and list of id of each documents
        """
        tokenize_title_lst, tokenize_content_lst, id_lst = self._tokenize_document_and_id()
        tokenized_document_corpus: list[list[str]] = []
        for i in range(len(tokenize_title_lst)):
            tokenize_title = tokenize_title_lst[i]
            tokenize_content = tokenize_content_lst[i]
            tokenize_document = tokenize_title + tokenize_content
            tokenized_document_corpus.append(tokenize_document)
        bm25plus = BM25Plus(tokenized_document_corpus)
        return bm25plus, id_lst

    def _tokenize_document_and_id(self) -> tuple[list[list[str]], list[list[str]], list[str]]:
        """
        Tokenize the title, content and take the id of each document in the dataset

        Returns:
            tuple[list[list[str]],list[list[str]],list[str]]: Tuple contains list of tokenized titles,\
            list of tokenized contents, and list of ids for each document.\
            The index is correspond to each other, which mean tokenize_title_lst[i]\
            is the title of the document whose content is tokenize_content_lst[i].\
            Similar to id as well.
        """
        tokenize_title_lst: list[list[str]] = []
        tokenize_content_lst: list[list[str]] = []
        id_lst: list[str] = []
        for _, _, tokenize_title, tokenize_content, id, _, _ in self.document_dataset:
            tokenize_title_lst.append(tokenize_title)
            tokenize_content_lst.append(tokenize_content)
            id_lst.append(id)
        return tokenize_title_lst, tokenize_content_lst, id_lst

    def get_documents_ranking(self, tokenized_query: list[str]) -> list[tuple[str, float]]:
        """
        Get the list of lexical matching score between query and each document in dataset.

        Returns:
            list[tuple[str,float]]: List contains pairs of document's id and its matching score to the query.\
            This list is sorted in descending order of matching score.\
            Depending on the mode is production or training, there will be a limit to the number of pairs returned or not.
        """
        matching_scores = self.bm25plus.get_scores(tokenized_query)
        top_relevant_index_set = argsort(matching_scores)[::-1]
        id_relevant_score_pairs: list[tuple[str, float]] = []
        for index in top_relevant_index_set:
            relevant_score_raw: float64 = matching_scores[index]
            relevant_score: float = relevant_score_raw.item()
            id: str = self.id_lst[index]
            pair: tuple[str, float] = (id, relevant_score)
            id_relevant_score_pairs.append(pair)

        return id_relevant_score_pairs
