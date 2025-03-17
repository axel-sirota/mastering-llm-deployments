"""
Lab 2: Fine-Tuning for Sentiment Analysis with DistilBERT

This script demonstrates a basic fine-tuning workflow for sentiment analysis using a local copy
of "distilbert-base-uncased-finetuned-sst-2-english". It attempts to load the IMDb dataset from disk.
If unavailable, it falls back to a dummy dataset.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk, Dataset

def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

def main():
    model_name = "local_models/distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, local_files_only=True)
    model = None # [EXERCISE] Add any parameters you think are necessary.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    try:
        dataset = None # [EXERCISE] Add any parameters you think are necessary.
        train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    except Exception as e:
        print(f"Failed to load IMDb dataset from disk: {e}")
        data = {
            "text": [
                "This is a very bad product. I am extremely disappointed.",
                "I absolutely love this! It works great and I am very happy."
            ],
            "label": [0, 1]
        }
        train_dataset = Dataset.from_dict(data)
    
    tokenized_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2
        # [EXERCISE] Add any parameters you think are necessary.
    )
    
    trainer = Trainer(
        model=model,
        # [EXERCISE] Add any parameters you think are necessary.
    )
    
    print("Starting fine-tuning for sentiment analysis...")
    trainer.train()
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
