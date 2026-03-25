"""
Lab 1b: Basic Fine-Tuning for Sentiment Analysis with DistilBERT

This script demonstrates a basic fine-tuning workflow for sentiment analysis using a local copy
of "distilbert-base-uncased-finetuned-sst-2-english". [EXERCISE]
Below, fill in the missing training implementation.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], truncation=True)

def main():
    model_name = "local_models/distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Create a dummy dataset
    data = {
        "text": [
            "This is a very bad product. I am extremely disappointed.",
            "I absolutely love this! It works great and I am very happy."
        ],
        "label": [0, 1]
    }
    dataset = Dataset.from_dict(data)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting fine-tuning for sentiment analysis...")
    trainer.train()
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
