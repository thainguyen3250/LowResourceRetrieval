from torch.utils.data import Dataset
from transformers import BatchEncoding
from .document_dataset import DocumentDataset
from ..language_processing.language_processing import LanguageProcessing
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast



class MLMFineTuneDataset(Dataset):
    """
    Class for representing the MLM-style samples dataset. This class uses preprocess documents
    getting from the DocumentDataset instance to generate correspond MLM-style samples.
    """

    def __init__(self, document_dataset: DocumentDataset) -> None:
        """
        Args:
            document_dataset (DocumentDataset): An object of DocumentDataset class, represent a document dataset use to generate MLM samples
        """
        self.document_dataset = document_dataset
        self.samples: list[dict] = self._create_samples()

    def _create_samples(self) -> list[dict]:
        """
        Generate the list of MLM-style samples from each document in the dataset of document provided in constructor.

        Returns:
            list[dict]: List of dictionary, each of dictionary is a MLM-style sample with two keys: 
            `input_ids` represent the sequence of ids of a chunk of token, and 
            `attention_mask` represent the mask of each corresponding token (1 is normal token, 0 is padding).
        """
        samples: list[dict] = []
        for _, _, _, tokenize_content, _, _, _ in self.document_dataset:
            document_language_processing: PreTrainedTokenizer | PreTrainedTokenizerFast = \
                self.document_dataset.language_processing.pre_trained_tokenizer_model()
            model_max_length: int = document_language_processing.model_max_length
            stride: int = min(model_max_length // 10, 50)
            example_lst: BatchEncoding = document_language_processing(
                tokenize_content,
                is_split_into_words=True,
                return_tensors='pt',
                max_length=model_max_length,
                truncation=True,
                stride=stride,
                padding='max_length')
            for i in range(len(example_lst["input_ids"])):
                example = {
                    "input_ids": example_lst["input_ids"][i],
                    "attention_mask": example_lst["attention_mask"][i]
                }
                samples.append(example)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
