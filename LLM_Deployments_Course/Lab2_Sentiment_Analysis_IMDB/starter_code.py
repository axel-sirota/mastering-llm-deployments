"""
Lab 2: Fine-Tuning for Sentiment Analysis with IMDb using LoRA

This script demonstrates how to fine-tune the "local_models/flan-t5-xl" model on the IMDb dataset using LoRA.
The task is formulated as a text-to-text problem:
Input: "Classify sentiment: <review text>"
Target: "positive" or "negative"
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_from_disk, load_dataset
from peft import get_peft_model, LoraConfig

def tokenize_function(example, tokenizer):
    example["input_text"] = "Classify sentiment: " + example["text"]
    model_inputs = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=128)
    target = example["label_text"]
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target, truncation=True, padding="max_length", max_length=16)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    model_name = "local_models/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    print("LoRA adapter applied to the model.")
    dataset = load_from_disk("local_datasets/imdb")
    train_dataset = dataset["train"].shuffle(seed=42).select(range(5000))
    def map_labels(example):
        example["label_text"] = "positive" if example["label"] == 1 else "negative"
        return example
    train_dataset = train_dataset.map(map_labels)
    tokenized_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer).to("cuda" if torch.cuda.is_available() else "cpu"), batched=True)
    training_args = TrainingArguments(
        output_dir="./sentiment_results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        logging_steps=50,
        save_steps=100,
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
