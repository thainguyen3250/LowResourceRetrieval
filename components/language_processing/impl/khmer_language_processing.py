from functools import reduce
import os
import re
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Callable
from ...language_processing.language_processing import LanguageProcessing

class KhmerLanguageProcessing(LanguageProcessing):
    _instance = None
    _word_segment = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(KhmerLanguageProcessing, cls).__new__(cls)
        return cls._instance

    def __init__(
            self, 
            pre_trained_tokenizer_model: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
            tokenizer: Callable[[str], list[str]] | None = None,
            encoder: Callable[[str | list[str]], list[int]] | None = None
    ):
        self._pre_trained_tokenizer_model = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base") if \
            pre_trained_tokenizer_model is None else pre_trained_tokenizer_model
        self._text_preprocessing = self._load_text_preprocessing()
        self._tokenizer = self._pre_trained_tokenizer_model.tokenize if tokenizer is None else tokenizer
        self._encoder = self._pre_trained_tokenizer_model.encode if encoder is None else encoder
    
    def _load_text_preprocessing(self):
        def split_khmer_sentences(text):
            # Define sentence-ending punctuation, ensuring "។ល។" is treated as a unit
            sentence_endings = r"(។ល។|៕|។|\?|!)"

            # Use regex to split while keeping the punctuation attached
            sentences = re.split(sentence_endings, text)

            # Merge punctuation back with its respective sentence
            result = []
            for i in range(0, len(sentences) - 1, 2):
                sentence = sentences[i].strip() + sentences[i + 1]  # Combine with punctuation
                result.append(sentence)

            # Handle potential last sentence without punctuation
            if len(sentences) % 2 != 0 and sentences[-1].strip():
                result.append(sentences[-1].strip())

            return result
        
        return split_khmer_sentences
    
    def text_preprocessing(self, text):
        return self._text_preprocessing(text)
        
    def pre_trained_tokenizer_model(self):
        return self._pre_trained_tokenizer_model
    
    def tokenizer(self, text: str):
        return self._tokenizer(text)
    
    def encoder(self, text: str | list[str]):
        return self._encoder(text)