import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, load_from_disk

# --- Setup device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Load DialogSum dataset (using a small subset for demo purposes) ---
try:
    dataset = load_from_disk("local_datasets/dialogsum")
    dataset = dataset["train"].select(range(50))
except Exception:
    dataset = load_dataset("knkarthick/dialogsum", split="train[:50]")

# --- Load teacher model and tokenizer (FLAN-T5-XL) ---
teacher_model = AutoModelForSeq2SeqLM.from_pretrained("local_models/flan-t5-xl", local_files_only=True).to(device)
teacher_tokenizer = AutoTokenizer.from_pretrained("local_models/flan-t5-xl", local_files_only=True)
teacher_model.eval()  # set teacher to evaluation mode

# --- Function to generate teacher summary for each dialogue ---
def generate_teacher_summary(example):
    # Tokenize the dialogue for the teacher model
    inputs = teacher_tokenizer(
        example["dialogue"],
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs.input_ids.to(device)
    
    # Generate a summary using the teacher model (using beam search for better quality)
    summary_ids = teacher_model.generate(input_ids, max_length=128, num_beams=4)
    
    # Decode the generated ids to text
    summary = teacher_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    example["teacher_summary"] = summary
    return example

# --- Apply teacher generation to the dataset ---
# This adds a new field "teacher_summary" to each example.
dataset_with_teacher = dataset.map(generate_teacher_summary)

# --- Prepare training data for the student model ---
# For this example, we'll fine-tune T5-small using the teacher-generated summaries.
student_tokenizer = AutoTokenizer.from_pretrained("local_models/t5-small", local_files_only=True)

def tokenize_data(example):
    # Optionally, add a task-specific prefix. Here we use "summarize:".
    input_text = "summarize: " + example["dialogue"]
    target_text = example["teacher_summary"]

    # Tokenize the input and target (teacher summary)
    model_inputs = student_tokenizer(
        input_text,
        text_target=target_text,
        max_length=512,
        truncation=True,
    )
    # model_inputs["labels"] is automatically set by text_target=
    return model_inputs

# Map the tokenization over the new dataset
tokenized_dataset = dataset_with_teacher.map(tokenize_data, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# --- Load the student model (T5-small) ---
student_model = AutoModelForSeq2SeqLM.from_pretrained("local_models/t5-small", local_files_only=True).to(device)

# --- Set up the Trainer ---
training_args = TrainingArguments(
    output_dir="./student_distilled",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    report_to="none",
    # use_cpu=False is the default; omit entirely to use GPU if available
)

trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# --- Begin training ---
trainer.train()
