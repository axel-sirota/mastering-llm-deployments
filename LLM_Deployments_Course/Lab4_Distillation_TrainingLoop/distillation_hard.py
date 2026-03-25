import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk, load_dataset
from torch.optim import AdamW
import torch.nn.functional as F

def tokenize_function(example, tokenizer, max_input_length=256, max_target_length=128):
    # Tokenize the dialogue (input) and the reference summary (target)
    model_inputs = tokenizer(
        example["dialogue"],
        text_target=example["summary"],
        truncation=True,
        padding="max_length",
        max_length=max_input_length,
    )
    # model_inputs["labels"] automatically set by text_target=
    # Convert to tensor for manual training loop
    model_inputs["labels"] = torch.tensor(model_inputs["labels"])
    return model_inputs

def generate_text(model, tokenizer, input_text, device, max_length=128, num_beams=4):
    """
    This function adds generation capability to the distilled student model.
    It uses the built-in generate method (available in T5ForConditionalGeneration) to produce text.
    """
    model.eval()
    # Optionally, prepend a task-specific prefix (here "summarize:") to the input text.
    input_prompt = "summarize: " + input_text
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return output_text

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Load teacher (FLAN-T5-XL) ---
    teacher_model_name = "local_models/flan-t5-xl"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, local_files_only=True)
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_name, local_files_only=True).to(device)
    teacher_model.eval()

    # --- Load student (T5-small) ---
    student_model_name = "local_models/t5-small"
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, local_files_only=True)
    student_model = AutoModelForSeq2SeqLM.from_pretrained(student_model_name, local_files_only=True).to(device)
    student_model.train()

    # --- Load DialogSum dataset ---
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
    # Shuffle and select a subset for training
    train_dataset = dataset.shuffle(seed=42).select(range(100))
    tokenized_dataset = [tokenize_function(x, teacher_tokenizer) for x in train_dataset]

    # --- Set up optimizer ---
    optimizer = AdamW(student_model.parameters(), lr=3e-5)
    num_epochs = 1
    batch_size = 4
    print("Starting distillation training loop...")

    # --- Distillation Training Loop ---
    for epoch in range(num_epochs):
        for i in range(0, len(tokenized_dataset), batch_size):
            batch = tokenized_dataset[i:i+batch_size]
            input_ids = torch.stack([torch.tensor(b["input_ids"]) for b in batch]).to(device)
            attention_mask = torch.stack([torch.tensor(b["attention_mask"]) for b in batch]).to(device)
            labels = torch.stack([b["labels"] for b in batch]).to(device)
            
            # Get teacher outputs (no gradients needed)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                teacher_logits = teacher_outputs.logits
            
            # Get student outputs
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            student_logits = student_outputs.logits
            
            # Compute KL Divergence loss between student and teacher distributions
            loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean"
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 8 == 0:
                print(f"Epoch {epoch+1}, Batch {i//batch_size+1}, Loss: {loss.item():.4f}")
    print("Distillation training complete!")

    # --- Generation with the Distilled Student Model ---
    # After distillation, we add the capability to generate text.
    # Since we're using T5ForConditionalGeneration for the student, it already has a generate method.
    student_model.eval()
    example_dialogue = ("Speaker1: Can you tell me about the weather forecast? "
                        "Speaker2: It is supposed to be sunny tomorrow.")
    generated_summary = generate_text(
        student_model, student_tokenizer, example_dialogue, device, max_length=128, num_beams=4
    )
    print("\nGenerated Summary from distilled student model:")
    print(generated_summary)
    
if __name__ == "__main__":
    main()
