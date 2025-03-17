"""
Lab 4: Efficient Distillation using a Custom Training Loop

This script distills knowledge from a teacher model (local_models/flan-t5-xl) into a smaller student model (t5-small)
using a custom training loop.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk, load_dataset
from torch.optim import AdamW
import torch.nn.functional as F

def tokenize_function(example, tokenizer, max_input_length=256, max_target_length=128):
    model_inputs = tokenizer(example["dialogue"], truncation=True, padding="max_length", max_length=max_input_length)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["summary"], truncation=True, padding="max_length", max_length=max_target_length)
    model_inputs["labels"] = torch.tensor(labels["input_ids"])
    return model_inputs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model_name = "local_models/flan-t5-xl"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_name).to(device)
    teacher_model.eval()
    student_model_name = "t5-small"
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    student_model = AutoModelForSeq2SeqLM.from_pretrained(student_model_name).to(device)
    student_model.train()
    try:
        dataset = load_from_disk("local_datasets/dialogsum")
    except Exception as e:
        print("DialogSum dataset not found. Using dummy data.")
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
        dataset = Dataset.from_dict(data)
    train_dataset = dataset.shuffle(seed=42).select(range(100))
    tokenized_dataset = [tokenize_function(x, teacher_tokenizer) for x in train_dataset]
    optimizer = AdamW(student_model.parameters(), lr=3e-5)
    num_epochs = 1
    batch_size = 4
    print("Starting distillation training loop...")
    for epoch in range(num_epochs):
        for i in range(0, len(tokenized_dataset), batch_size):
            batch = tokenized_dataset[i:i+batch_size]
            input_ids = torch.stack([torch.tensor(b["input_ids"]) for b in batch]).to(device)
            attention_mask = torch.stack([torch.tensor(b["attention_mask"]) for b in batch]).to(device)
            labels = torch.stack([b["labels"] for b in batch]).to(device)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                teacher_logits = teacher_outputs.logits
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            student_logits = student_outputs.logits
            loss = F.kl_div(F.log_softmax(student_logits, dim=-1),
                            F.softmax(teacher_logits, dim=-1),
                            reduction="batchmean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 8 == 0:
                print(f"Epoch {epoch+1}, Batch {i//batch_size+1}, Loss: {loss.item():.4f}")
    print("Distillation training complete!")
    
if __name__ == "__main__":
    main()
