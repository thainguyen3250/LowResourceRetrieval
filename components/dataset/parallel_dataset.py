from functools import reduce
import os
from torch.utils.data import Dataset
import csv
from ..language_processing.language_processing import LanguageProcessing


class ParallelDataset(Dataset):
    def __init__(
        self,
        parallel_dir: str,
        teacher_language_processing: LanguageProcessing,
        student_language_processing: LanguageProcessing
    ):
        """
        Args:
            parallel_dir (str): Path to the folder containing CSV files with parallel sentences.

            teacher_language_processing (LanguageProcessing): Language processing object\
            for the teacher language.

            student_language_processing (LanguageProcessing): Language processing object\
            for the student language.
        """
        self.parallel_dir: str = parallel_dir
        self.teacher_language_processing: LanguageProcessing = teacher_language_processing,
        self.student_language_processing: LanguageProcessing = student_language_processing,
        self.pairs = self._load_pairs()

    def _load_pairs(self) -> list[tuple[list[str], list[str]]]:
        """
        Load parallel sentence pairs from CSV files in the specified folder.
        Then, word-segment the two sentences in each pair.

        Returns:
            list: A list of tuples containing pairs of word-segmented sentences.
        """
        pairs: list[tuple[list[str], list[str]]] = []
        for file_name in os.listdir(self.parallel_dir):
            if file_name.endswith('.csv'):  # Process only CSV files
                file_path = os.path.join(self.parallel_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        if len(row) == 2:  # Ensure there are exactly two sentences
                            # Append the pair of sentences
                            teacher_language_sentence: str = row[0].strip()
                            student_language_sentence: str = row[1].strip()
                            tokenized_teacher_language_sentence: list[str] = self.teacher_language_processing.tokenizer(
                                teacher_language_sentence)
                            tokenized_student_language_sentence: list[str] = self.student_language_processing.tokenizer(
                                student_language_sentence)
                            pairs.append(
                                (tokenized_teacher_language_sentence, tokenized_student_language_sentence))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[list[str], list[str]]:
        tokenized_teacher_language_sentence, tokenized_student_language_sentence = self.pairs[idx]
        return tokenized_teacher_language_sentence, tokenized_student_language_sentence