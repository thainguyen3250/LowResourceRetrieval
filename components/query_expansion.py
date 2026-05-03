import concurrent.futures
import numpy as np
from numpy import float64, argsort
from rank_bm25 import BM25Plus
from typing import TypeAlias, Literal, Dict, List, Set, Tuple
from functools import lru_cache
from math import log, inf
import threading
from .dataset import DocumentDataset

SourceForExpansion: TypeAlias = Literal[
    'COLLECTION_SET_TITLE',
    'COLLECTION_SET_CONTENT',
    'RELEVANT_SET_TITLE',
    'RELEVANT_SET_CONTENT'
]

class QueryExpansion:
    # Too high will affect the speed, especially if the document is long
    # The paper uses this number in experiment
    LIMIT_K_DOCS_FOR_RELEVANT_SET = 10

    # The paper also uses this number in experiment
    NUMBER_OF_EXPANSION_TERM = 30

    THRESHOLD_FOR_EM_ALGO = 0.000001
    
    def __init__(self, document_dataset: DocumentDataset) -> None:
        """
        Args:
            document_dataset: An object of DocumentDataset class, represent a document dataset use to expand the query
        """
        self.document_dataset: DocumentDataset = document_dataset
        self.sources: dict[SourceForExpansion, list[list[str]]] = {
            'COLLECTION_SET_TITLE': [],
            'COLLECTION_SET_CONTENT': [],
            'RELEVANT_SET_TITLE': [],
            'RELEVANT_SET_CONTENT': []
        }
        self.sources['COLLECTION_SET_TITLE'], self.sources['COLLECTION_SET_CONTENT'] = self._get_tokenize_document()
        
        # Perform collection set calculation once and store as a list for faster access
        self.collection_set: set[str] = self._get_collection_set()
        self.collection_set_size = len(self.collection_set)
        
        # Default probability value
        self.default_prob = 1.0 / float(self.collection_set_size)
        
        self.bm25plus: BM25Plus = self._load_bm25plus()
        
        # At first, this dict is empty, so we need to use .get() method with default value when retrieving prob,
        # so that if key does not exist yet, default value will be return.
        # Default value will be 1.0 / float(len(self.collection_set)), which mean it follows the uniform distribution.
        self.prob_expansion_term_represents_source: dict[tuple[str, SourceForExpansion], float] = {}

        self.prob_of_selecting_source: dict[SourceForExpansion, float] = {
            # Initialize all 4 probs equally. Maximization step will update these probs later.
            'COLLECTION_SET_TITLE': 0.25,
            'COLLECTION_SET_CONTENT': 0.25,
            'RELEVANT_SET_TITLE': 0.25,
            'RELEVANT_SET_CONTENT': 0.25
        }
        
        # Like above, this dict is also empty and required to be treated similarly
        self.prob_term_belongs_to_source: dict[tuple[str, SourceForExpansion], float] = {}

    def _get_tokenize_document(self) -> tuple[list[list[str]], list[list[str]]]:
        """
        Get the list of tokenized title and content of all documents in the dataset.
        
        Returns:
            tuple[list[list[str]],list[list[str]]]: Tuple contains list of tokenized titles and list of tokenized contents.\
            The index is correspond to each other, which mean tokenize_title_lst[i] is the title of the document whose content is tokenize_content_lst[i].
        """
        tokenize_title_lst: list[list[str]] = []
        tokenize_content_lst: list[list[str]] = []
        for _, _, tokenize_title, tokenize_content, _, _, _ in self.document_dataset:
            tokenize_title_lst.append(tokenize_title)
            tokenize_content_lst.append(tokenize_content)
        return tokenize_title_lst, tokenize_content_lst

    def _get_collection_set(self) -> set[str]:
        """
        Get all terms from the collection set. This will get term set from two collection set (title and content),
        then combine to form the final term set.

        Returns:
            set[str]: The set contain all terms from collection set.
        """
        collection_set: set[str] = self._get_term_set_of_source("COLLECTION_SET_TITLE")
        collection_set.update(self._get_term_set_of_source("COLLECTION_SET_CONTENT"))
        return collection_set
    
    def _get_term_set_of_source(self, source: SourceForExpansion) -> set[str]:
        """
        Get all terms from the specify source.

        Args:
            source (SourceForExpansion): Name of the source that needs to get terms

        Returns:
            set[str]: The set contain all terms extract from all document in the specify source.
        """
        term_set: set[str] = set()
        for sequence in self.sources[source]:
            for term in sequence:
                term_set.add(term)
        return term_set

    def _load_bm25plus(self) -> BM25Plus:
        """
        Load BM25Plus instance initialized with documents from the document dataset.

        Returns:
            BM25Plus: BM25Plus instance initialized with documents from the document dataset
        """
        tokenized_document_corpus: list[list[str]] = []
        for i in range(len(self.sources['COLLECTION_SET_TITLE'])):
            tokenize_title: list[str] = self.sources['COLLECTION_SET_TITLE'][i]
            tokenize_content: list[str] = self.sources['COLLECTION_SET_CONTENT'][i]
            tokenize_document: list[str] = tokenize_title + tokenize_content
            tokenized_document_corpus.append(tokenize_document)
        bm25plus = BM25Plus(tokenized_document_corpus)
        return bm25plus

    def get_expansion_term(self, tokenized_query: list[str]) -> list[str]:
        """
        Get the list of tokenized expansion terms for the input query.
        This is also the only public method for this class, the only method you need.

        Args:
            tokenized_query (list[str]): Query which has been tokenized with the same tokenizer used for the documents

        Returns:
            list[str]: The list of tokenized expansion terms
        """
        self._retrive(tokenized_query)
        return self._expand(tokenized_query)

    def _retrive(self, tokenized_query: list[str]) -> None:
        """
        Get the list of the top highest lexical similarity documents for a query and add these documents to relevant set.
        This method will also call helper method to clear out the relevant set of the previous query, 
        and reset all probabiblities before processing current query.

        Args:
            tokenized_query (list[str]): Query which has been tokenized with the same tokenizer used for the documents
        """
        self._clear_previous_result()

        matching_scores = self.bm25plus.get_scores(tokenized_query)
        top_relevant_index_set = argsort(matching_scores)[::-1][:self.LIMIT_K_DOCS_FOR_RELEVANT_SET]

        for index in top_relevant_index_set:
            self.sources['RELEVANT_SET_TITLE'].append(
                self.sources['COLLECTION_SET_TITLE'][index])
            self.sources['RELEVANT_SET_CONTENT'].append(
                self.sources['COLLECTION_SET_CONTENT'][index])

    def _clear_previous_result(self) -> None:
        """
        Helper method to clear out the relevant set of the previous query, and reset all probabiblities.
        """
        self.sources['RELEVANT_SET_TITLE'].clear()
        self.sources['RELEVANT_SET_CONTENT'].clear()
        self.prob_of_selecting_source = {
            'COLLECTION_SET_TITLE': 0.25,
            'COLLECTION_SET_CONTENT': 0.25,
            'RELEVANT_SET_TITLE': 0.25,
            'RELEVANT_SET_CONTENT': 0.25
        }
        self.prob_expansion_term_represents_source.clear()
        self.prob_term_belongs_to_source.clear()

    def _log_likelihood(self, observation_sequence: set[str]) -> float:
        """
        Calculate the log likelihood value for the observation sequence.

        Args:
            observation_sequence (set[str]): The term set of the source that is considered to be observation sequence.\
            For ease of understanding, this term set is all the words from relevant set (title) or relevant set (content),
            depends on which relevant set you are calculating.

        Returns:
            float: The log likelihood value calculated
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for term in observation_sequence:
                futures.append(executor.submit(self._compute_term_likelihood, term))
            
            likelihoods = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return sum(likelihoods)
    
    def _compute_term_likelihood(self, term):
        """Helper function to compute likelihood for a single term"""
        accumulate_likelihood_of_source: float = 0.0
        for source in self.sources.keys():
            term_source_pair: tuple[str, SourceForExpansion] = (term, source)
            prob_term_belongs_to_source: float = self.prob_term_belongs_to_source.get(term_source_pair, 0.25)
            prob_of_selecting_source: float = self.prob_of_selecting_source[source]

            accumulate_likelihood_of_collection_set: float = log(prob_of_selecting_source)
            if term in self.collection_set:
                prob_expansion_term_represents_source: float = self.prob_expansion_term_represents_source.get(
                    term_source_pair, self.default_prob)
                if prob_expansion_term_represents_source > 0: 
                    accumulate_likelihood_of_collection_set += log(prob_expansion_term_represents_source)
            
            accumulate_likelihood_of_source += prob_term_belongs_to_source * accumulate_likelihood_of_collection_set
            
        return accumulate_likelihood_of_source

    def _maximization_step(self, observation_sequence: set) -> None:
        """
        Maximization step in EM algorithm. This will calculate new probability of selecting a source and
        probability an expansion term represents a source, and then save them.

        Args:
            observation_sequence (set[str]): The term set of the source that is considered to be observation sequence.\ 
            For ease of understanding, this term set is all the words from relevant set (title) or relevant set (content),
            depends on which relevant set you are calculating.
        """
        updated_prob_of_selecting_source: dict[SourceForExpansion, float] = {}
        updated_prob_expansion_term_represents_source: dict[tuple[str, SourceForExpansion], float] = {}

        # Parallelizing calculations of probability of choosing each source
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            for source in self.sources.keys():
                futures.append(executor.submit(self._maximize_prob_of_selecting_source, source, observation_sequence))

            # Collect the results
            prob_of_selecting_source_numerators: list[tuple[SourceForExpansion, float]] = \
                [future.result() for future in concurrent.futures.as_completed(futures)]

            # Normalize the results
            denominator = sum([numerator for _, numerator in prob_of_selecting_source_numerators])
            for source_to_maximize, numerator in prob_of_selecting_source_numerators:
                prob = numerator / denominator
                updated_prob_of_selecting_source[source_to_maximize] = prob

        for source in self.sources.keys():
            # Parallelizing calculations of probability of each expansion term representing the source
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []

                for expansion_term_to_maximize in self.collection_set:
                    futures.append(executor.submit(self._maximize_prob_expansion_term_represents_source, 
                                                   source, expansion_term_to_maximize, observation_sequence))
                
                # Collect the results
                prob_expansion_term_represents_source_numerators: list[tuple[str, float]] = \
                    [future.result() for future in concurrent.futures.as_completed(futures)]

                denominator = sum([numerator for _, numerator in prob_expansion_term_represents_source_numerators])
                for expansion_term_to_maximize, numerator in prob_expansion_term_represents_source_numerators:
                    prob = numerator / denominator
                    updated_prob_of_selecting_source[(expansion_term_to_maximize, source)] = prob

        self.prob_of_selecting_source = updated_prob_of_selecting_source
        self.prob_expansion_term_represents_source = updated_prob_expansion_term_represents_source

    def _maximize_prob_of_selecting_source(
            self,
            source_to_maximize: SourceForExpansion,
            observation_sequence: set[str]
    ) -> tuple[SourceForExpansion, float]:
        """
        Calculate the new probability of selecting a source.

        Args:
            source_to_maximize (SourceForExpansion): Name of the source you want to calculate new probability

            observation_sequence (set[str]): The term set of the source that is considered to be observation sequence.\
            For ease of understanding, this term set is all the words from relevant set (title) or relevant set (content),
            depends on which relevant set you are calculating.

        Returns:
            float: The new probability calculated.
        """
        numerator: float = 0.0
        
        for term in observation_sequence:
            term_source_to_maximize_pair: tuple[str, SourceForExpansion] = (term, source_to_maximize)
            numerator += self.prob_term_belongs_to_source.get(term_source_to_maximize_pair, 0.25)
        
        # The denominator is just the sum of all the numerators so no need for calculation of denominator
        return source_to_maximize, numerator

    def _maximize_prob_expansion_term_represents_source(
            self,
            source_to_maximize: SourceForExpansion,
            expansion_term_to_maximize: str,
            observation_sequence: set[str]
    ) -> tuple[str, float]:
        """
        Calculate the new probability an expansion term represents a source.

        Args:
            source_to_maximize (SourceForExpansion): Name of the source you want to calculate new probability

            expansion_term_to_maximize (str): The term you want to calculate new probability

            observation_sequence (set[str]): The term set of the source that is considered to be observation sequence.\ 
            For ease of understanding, this term set is all the words from relevant set (title) or relevant set (content),
            depends on which relevant set you are calculating.

        Returns:
            float: The new probability calculated.
        """
        # Quick check - if term isn't in observation sequence, can be more efficient
        if expansion_term_to_maximize not in observation_sequence:
            return expansion_term_to_maximize, 0.0
        
        term_source_to_maximize_pair: tuple[str, SourceForExpansion] = (expansion_term_to_maximize, source_to_maximize)
        numerator: float = self.prob_term_belongs_to_source.get(term_source_to_maximize_pair, 0.5)
        
        return expansion_term_to_maximize, numerator

    def _estimation_step(self, observation_sequence: set) -> None:
        """
        Expectation step in EM algorithm. This will calculate new probability that a term belongs to a source, 
        and then save it

        Args:
            observation_sequence (set[str]): The term set of the source that is considered to be observation sequence.\ 
            For ease of understanding, this term set is all the words from relevant set (title) or relevant set (content),
            depends on which relevant set you are calculating.
        """
        updated_prob_term_belongs_to_source: dict[tuple[str, SourceForExpansion], float] = {}
        
        # Process in parallel with optimized batching
        for term in observation_sequence:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []

                # Create all pairs for processing
                pairs: list[tuple[str, SourceForExpansion]] = [(term, source) for source in self.sources.keys()]
                for term, source in pairs:
                    futures.append(executor.submit(self._estimate_prob_term_belongs_to_source, term, source))

                # Collect the results
                numerators: list[tuple[SourceForExpansion, float]] = \
                    [future.result() for future in concurrent.futures.as_completed(futures)]

                # Normalize the results
                denominator = sum([numerator for _, numerator in numerators])
                for source_to_estimate, numerator in numerators:
                    prob = numerator / denominator
                    updated_prob_term_belongs_to_source[(term, source_to_estimate)] = prob
    
        self.prob_term_belongs_to_source = updated_prob_term_belongs_to_source

    def _estimate_prob_term_belongs_to_source(
            self,
            term_to_estimate: str,
            source_to_estimate: SourceForExpansion
    ) -> tuple[SourceForExpansion, float]:
        """
        Calculate new probability that a term belongs to a source.

        Args:
            term_to_estimate (str): The term you want to calculate new probability

            source_to_estimate (SourceForExpansion): Name of the source you want to calculate new probability

        Returns:
            float: The new probability calculated.
        """
        prob: float = 1.0
        if term_to_estimate in self.collection_set:
            expansion_term_source_to_estimate_pair: tuple[str, SourceForExpansion] = (term_to_estimate, source_to_estimate)
            prob *= self.prob_expansion_term_represents_source.get(
                expansion_term_source_to_estimate_pair, self.default_prob)

        numerator: float = self.prob_of_selecting_source[source_to_estimate] * prob
    
        return source_to_estimate, numerator
        

    def _perform_em_algorithm(
        self, 
        observation_sequence: set[str], 
        source: SourceForExpansion, 
        tokenized_query: list[str]
    ) -> list[tuple[str, float]]:
        """
        Perform the Expectation-Maximization (EM) algorithm to estimate the probabilities of expansion terms.
        Args:
            observation_sequence (set[str]): A set of observed terms.
            source (SourceForExpansion): The source from which expansion terms are derived.
            tokenized_query (list[str]): The original query tokenized into a list of terms.
        Returns:
            list[tuple[str,float]]: A list of tuples where each tuple contains an expansion term and its corresponding probability, sorted in descending order of probability.
        """
        previous_likelihood = -inf
        current_likelihood = +inf
        iteration = 0
        max_iterations = 30  # Add a cap on iterations to prevent infinite loops
        
        while True:
            iteration += 1
            current_likelihood = self._log_likelihood(observation_sequence)
            
            # Convergence check
            if (current_likelihood <= previous_likelihood + self.THRESHOLD_FOR_EM_ALGO or 
                iteration >= max_iterations):
                break
                
            previous_likelihood = current_likelihood
            
            # print(f"Starting iteration {iteration} for source {source}")
            self._estimation_step(observation_sequence)     
            self._maximization_step(observation_sequence)

        expansion_prob_dict: dict[tuple[str, SourceForExpansion], float] = {
            k: v for k, v in self.prob_expansion_term_represents_source.items() if k[1] == source and k[0] not in tokenized_query}
        sorted_expansion_prob: list[tuple[tuple[str, SourceForExpansion], float]] = sorted(
            expansion_prob_dict.items(), key=lambda item: item[1], reverse=True)
        return [(term[0][0], term[1]) for term in sorted_expansion_prob[:self.NUMBER_OF_EXPANSION_TERM]]

    def _expand(self, tokenized_query: list[str]) -> list[str]:
        """
        Get the list of tokenized expansion terms by performing EM algorithm for each observation sequence:
        relevant set (title) and relevant set (content), then combining their expansion term list to form final list.

        Returns:
            list[str]: Final list of tokenized expansion terms
        """
        observation_sequence_title: set[str] = self._get_term_set_of_source("RELEVANT_SET_TITLE")
        observation_sequence_content: set[str] = self._get_term_set_of_source("RELEVANT_SET_CONTENT")

        # print("Starting EM algorithm for title set")
        expansion_term_with_prob_from_title_relevant_set = self._perform_em_algorithm(
            observation_sequence_title, "RELEVANT_SET_TITLE", tokenized_query)
        # print("DONE with title set")
        
        # print("Starting EM algorithm for content set")
        expansion_term_with_prob_from_content_relevant_set = self._perform_em_algorithm(
            observation_sequence_content, "RELEVANT_SET_CONTENT", tokenized_query)
        # print("DONE with content set")
        
        combined_expansion_terms: list[tuple[str, float]] = []
        term_prob_dict: dict[str, float] = {}

        for term, prob in expansion_term_with_prob_from_title_relevant_set:
            if term not in term_prob_dict or prob > term_prob_dict[term]:
                term_prob_dict[term] = prob

        for term, prob in expansion_term_with_prob_from_content_relevant_set:
            if term not in term_prob_dict or prob > term_prob_dict[term]:
                term_prob_dict[term] = prob

        combined_expansion_terms = list(term_prob_dict.items())

        sorted_expansion_term_final_prob = sorted(
            combined_expansion_terms, key=lambda item: item[1], reverse=True)

        expansion_term_final: list[str] = [
            term[0] for term in sorted_expansion_term_final_prob[:self.NUMBER_OF_EXPANSION_TERM]]
        return expansion_term_final