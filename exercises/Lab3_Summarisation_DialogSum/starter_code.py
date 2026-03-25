"""
Lab 3: Summarisation with DialogSum using LoRA

This script fine-tunes "flan-t5-xl" for summarisation on the DialogSum dataset using LoRA.
- Loads model and tokenizer from local files.
- The tokenize function sets both "labels" and "decoder_input_ids".
[EXERCISE] Please complete the training call.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_from_disk, Dataset
from peft import get_peft_model, LoraConfig

def tokenize_function(example, tokenizer):
    model_inputs = tokenizer(example["dialogue"], truncation=True, padding="max_length", max_length=256)
    labels = tokenizer(text_target=example["summary"], truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    model_name = "local_models/flan-t5-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    
    lora_config = LoraConfig(
        r=None,  # YOUR CODE: rank of the low-rank decomposition (integer, e.g. 16)
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=None,  # YOUR CODE: dropout probability for LoRA layers (float between 0 and 1)
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)
    print("LoRA adapter applied to the summarisation model.")
    
    try:
        dataset = load_from_disk("local_datasets/dialogsum")
        train_dataset = dataset["train"].shuffle(seed=42).select(range(500))
    except Exception as e:
        print(f"Failed to load DialogSum dataset from disk: {e}")
        data = {
            "dialogue": [
                "Speaker1: Hello, how are you? Speaker2: I am good, thank you. Speaker1: Great to hear!",
                "Speaker1: Can you help me with my order? Speaker2: Sure, what seems to be the issue?"
            ],
            "summary": [
                "A friendly greeting exchange.",
                "A customer seeks help with an order."
            ]
        }
        from datasets import Dataset
        train_dataset = Dataset.from_dict(data).shuffle(seed=42).select(range(500))
    
    tokenized_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    training_args = TrainingArguments(
        output_dir="./summarisation_results",
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
    
    print("Starting summarisation fine-tuning on DialogSum...")
    None  # YOUR CODE: call trainer.train() to start fine-tuning
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
