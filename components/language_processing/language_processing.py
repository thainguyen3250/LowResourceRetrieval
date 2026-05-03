from abc import ABC
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from abc import ABC, abstractmethod

class LanguageProcessing(ABC):
    """
    Abstract base class for language processing components.
    This class defines the interface for language processing components, including methods for
    tokenization, text preprocessing, and encoding. Implementations of this class should provide
    concrete implementations for these methods.
    """
    
    @abstractmethod
    def pre_trained_tokenizer_model(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """
        Get the pre-trained tokenizer model for the language processing component.
        Returns:
            PreTrainedTokenizer | PreTrainedTokenizerFast: The pre-trained tokenizer model
        """
        pass
    
    @abstractmethod
    def text_preprocessing(self, text: str) -> list[str]:
        """
        Preprocess the input text. This method should perform any necessary preprocessing steps,
        and also HAVE TO split the input text into a list of sentences.
        Args:
            text (str): The input text to preprocess.
        Returns:
            list[str]: The preprocessed text, split into a list of sentences.
        """
        pass

    @abstractmethod
    def tokenizer(self, text: str) -> list[str]:
        """
        Tokenize the input text. This method should tokenize the input text into a list of tokens.
        Args:
            text (str): The input text to tokenize.
        Returns:
            list[str]: The list of tokens produced by tokenizing the input text.
        """
        pass

    @abstractmethod
    def encoder(self, text: str | list[str]) -> list[int]:
        """
        Encode the input text. This method should encode the input text into a list of token ids.
        Args:
            text (str | list[str]): The input text to encode.
        Returns:
            list[int]: The list of token ids produced by encoding the input text.
        """
        pass
