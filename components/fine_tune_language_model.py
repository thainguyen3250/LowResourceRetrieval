from .dataset import DocumentDataset, MLMFineTuneDataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, AutoModelForMaskedLM


class FineTuneLanguageModel:
    def __init__(self, document_dataset: DocumentDataset, pretrained_model_name_or_path: str) -> None:
        """
        Args:
            document_dataset (DocumentDataset): An object of DocumentDataset class, represent a document dataset use to fine tune model

            pretrained_model_name_or_path (str): A string - the model id of a pretrained model hosted inside a model repo on huggingface.co.\
            OR a path to a directory containing model. This param will be used in `AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)`\
            to get the pre-trained model.
        """
        self.data_collator = DataCollatorForLanguageModeling(
            document_dataset.language_processing.pre_trained_tokenizer_model, mlm=True, mlm_probability=0.15)
        self.model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name_or_path)
        self.dataset: MLMFineTuneDataset = MLMFineTuneDataset(document_dataset)

    def train(self) -> str:
        """
        Create `TrainingArguments` instance and use it to fine tune the model automatically with `Trainer` API.

        Returns:
            str: Path to the directory containing the MLM fine-tuned model.
        """
        output_dir: str = f"/mlm_finetuned/{self.dataset.document_dataset.language}"
        training_args: TrainingArguments = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            gradient_accumulation_steps=4,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=20,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir='/mlm_finetuned/logs',
            logging_steps=500
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=self.data_collator,
        )
        trainer.train()
        return output_dir
