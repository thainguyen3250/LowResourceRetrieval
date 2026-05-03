from functools import reduce
import os
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Callable
from ...language_processing.language_processing import LanguageProcessing
import py_vncorenlp


class VietnameseLanguageProcessing(LanguageProcessing):
    _instance = None
    _word_segment = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VietnameseLanguageProcessing, cls).__new__(cls)
        return cls._instance
    
    def __init__(
            self, 
            pre_trained_tokenizer_model: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
            tokenizer: Callable[[str], list[str]] | None = None,
            encoder: Callable[[str | list[str]], list[int]] | None = None
    ):
        self._pre_trained_tokenizer_model = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", token=os.getenv("HUGGINGFACE_TOKEN")) if \
            pre_trained_tokenizer_model is None else pre_trained_tokenizer_model
        if self._word_segment is None:
            self._text_preprocessing = self._load_text_preprocessing()
        self._tokenizer = self._pre_trained_tokenizer_model.tokenize if tokenizer is None else tokenizer
        self._encoder = self._pre_trained_tokenizer_model.encode if encoder is None else encoder
    
    def _load_text_preprocessing(self):
        if os.path.isdir("/vncorenlp/models") == False or os.path.exists('/vncorenlp/VnCoreNLP-1.2.jar') == False:
            os.makedirs("/vncorenlp", exist_ok=True)
            from py_vncorenlp import download_model
            download_model("/vncorenlp")

        if VietnameseLanguageProcessing._word_segment is None:
            try:
                import py_vncorenlp
                from py_vncorenlp import VnCoreNLP
                VietnameseLanguageProcessing._word_segment = VnCoreNLP(
                    annotators=["wseg"], 
                    save_dir="/vncorenlp"
                ).word_segment
            except ValueError as e:
                if "VM is already running" in str(e):
                    import py_vncorenlp
                    from py_vncorenlp import VnCoreNLP
                    from jnius import autoclass
                    VietnameseLanguageProcessing._word_segment = VnCoreNLP(
                        annotators=["wseg"], 
                        save_dir="/vncorenlp",
                        skip_jvm_check=True 
                    ).word_segment
                else:
                    raise e
        return VietnameseLanguageProcessing._word_segment
    
    def text_preprocessing(self, text):
        return self._text_preprocessing(text)
        
    def pre_trained_tokenizer_model(self):
        return self._pre_trained_tokenizer_model
    
    def tokenizer(self, text: str):
        tokenized_text = self._text_preprocessing(text)
        tokenized_text = [tokenized_sent.split(' ') for tokenized_sent in tokenized_text]
        return [token.replace("_", " ") for tokenized_sent in tokenized_text for token in tokenized_sent]
    
    def encoder(self, text: str | list[str]):
        return self._encoder(text)