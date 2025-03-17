"""
Lab 3: Summarisation with DialogSum using LoRA

This script demonstrates fine-tuning "local_models/flan-t5-xl" for summarisation on the DialogSum dataset using LoRA.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_from_disk, load_dataset
from peft import get_peft_model, LoraConfig

def tokenize_function(example, tokenizer):
    model_inputs = tokenizer(example["dialogue"], truncation=True, padding="max_length", max_length=256)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["summary"], truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["decoder_input_ids"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels["attention_mask"]
    model_inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in model_inputs.items()}
    return model_inputs

def main():
    model_name = "local_models/flan-t5-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
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
    print("LoRA adapter applied to the summarisation model.")
    try:
        dataset = load_from_disk("local_datasets/dialogsum")
    except Exception as e:
        print("DialogSum dataset not found. Using dummy data.")
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
        dataset = Dataset.from_dict(data)
    if "train" in dataset:
        train_dataset = dataset["train"].shuffle(seed=42).select(range(500))
    else:
        train_dataset = dataset.shuffle(seed=42).select(range(500))
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
    trainer.train()
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
