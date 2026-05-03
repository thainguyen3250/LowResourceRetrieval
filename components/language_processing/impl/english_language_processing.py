import re
from typing import Callable
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from ...language_processing.language_processing import LanguageProcessing
import os


class EnglishLanguageProcessing(LanguageProcessing):
    _instance = None
    _word_segment = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EnglishLanguageProcessing, cls).__new__(cls)
        return cls._instance
    
    def __init__(
            self, 
            pre_trained_tokenizer_model: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
            tokenizer: Callable[[str], list[str]] | None = None,
            encoder: Callable[[str | list[str]], list[int]] | None = None
    ):
        self._pre_trained_tokenizer_model = AutoTokenizer.from_pretrained("bert-base-uncased", token=os.getenv("HUGGINGFACE_TOKEN")) if \
            pre_trained_tokenizer_model is None else pre_trained_tokenizer_model
        if self._word_segment is None:
            self._text_preprocessing = self._load_text_preprocessing()
        self._tokenizer = self._pre_trained_tokenizer_model.tokenize if tokenizer is None else tokenizer
        self._encoder = self._pre_trained_tokenizer_model.encode if encoder is None else encoder
    
    def _load_text_preprocessing(self):
        def split_english_sentences(text):
            return re.findall(r'\w+|[^\w\s]', text)

        return split_english_sentences
    
    def text_preprocessing(self, text):
        return self._text_preprocessing(text)
        
    def pre_trained_tokenizer_model(self):
        return self._pre_trained_tokenizer_model
    
    def tokenizer(self, text: str):
        return self._tokenizer(text)
    
    def encoder(self, text: str | list[str]):
        return self._encoder(text)