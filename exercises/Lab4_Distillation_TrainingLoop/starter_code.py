"""
Lab 4: Efficient Distillation using a Custom Training Loop

This script distills a teacher model (flan-t5-xl) into a student model (t5-small) using a custom training loop.
- Teacher is loaded from local files.
- Student is loaded normally.
- The tokenize function sets both "labels" and "decoder_input_ids".
[EXERCISE] Please implement the training loop to update the student model.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk, Dataset
from torch.optim import AdamW
import torch.nn.functional as F

def tokenize_function(example, tokenizer, max_input_length=256, max_target_length=128):
    model_inputs = tokenizer(example["dialogue"], truncation=True, padding="max_length", max_length=max_input_length)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["summary"], truncation=True, padding="max_length", max_length=max_target_length)
    model_inputs["labels"] = torch.tensor(labels["input_ids"])
    model_inputs["decoder_input_ids"] = torch.tensor(labels["input_ids"])
    return model_inputs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # [EXERCISE] Add any parameters you think are necessary. for student and teacher models
    
    try:
        dataset = load_from_disk("local_datasets/dialogsum")
        train_dataset = dataset.shuffle(seed=42).select(range(100))
    except Exception as e:
        print(f"Failed to load DialogSum dataset from disk: {e}")
        data = {
            "dialogue": [
                "Speaker1: Hello, how are you? Speaker2: I am good, thanks.",
                "Speaker1: What time is the meeting? Speaker2: It is at 3 PM."
            ],
            "summary": [
                "A greeting exchange.",
                "Meeting time information."
            ]
        }
        from datasets import Dataset
        train_dataset = Dataset.from_dict(data).shuffle(seed=42).select(range(100))
    
    tokenized_dataset = [tokenize_function(x, teacher_tokenizer) for x in train_dataset]
    
    optimizer = AdamW(student_model.parameters(), lr=3e-5)
    
    # [EXERCISE] Implement the training loop here:
    None  # please fill here with code to update the student model using KL divergence loss
    
    print("Distillation training complete!")
    
if __name__ == "__main__":
    main()
