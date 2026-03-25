import torch
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import torch.ao.quantization

def tokenize_function(example, tokenizer, max_input_length=256, max_target_length=128):
    # Add a prefix and tokenize the dialogue and its summary.
    input_text = "summarize: " + example["dialogue"]
    model_inputs = tokenizer(
        input_text,
        text_target=example["summary"],
        truncation=True,
        padding="max_length",
        max_length=max_input_length,
    )
    # Labels are automatically set as a list (not tensor) — correct for .map()
    # Do NOT call torch.tensor() here; set_format(type="torch") handles conversion
    return model_inputs

def train_qat():
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # --- Load dataset ---
    try:
        full_dataset = load_from_disk("local_datasets/dialogsum")
        dataset = full_dataset["train"].select(range(100))
    except Exception:
        dataset = load_dataset("knkarthick/dialogsum", split="train[:100]")
    tokenizer = AutoTokenizer.from_pretrained("local_models/flan-t5-small", local_files_only=True)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=False)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)
    
    # --- Load model ---
    model = AutoModelForSeq2SeqLM.from_pretrained("local_models/flan-t5-small", local_files_only=True)
    model.train()
    
    # --- QAT Preparation ---
    # Quantization configuration is only supported on CPU, so move model to CPU.
    model.to("cpu")
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
    # Disable quantization on embedding layers to avoid errors.
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.qconfig = None
    torch.ao.quantization.prepare_qat(model, inplace=True)
    
    # For training, we move the model back to GPU.
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5)
    
    # --- Training Loop ---
    num_epochs = 1
    print("Starting QAT training loop...")
    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    print("QAT training complete!")
    
    # --- Convert to Quantized Model ---
    # Conversion must happen on CPU.
    model.to("cpu")
    quantized_model = torch.ao.quantization.convert(model.eval(), inplace=False)
    print("Model converted to quantized version.")
    return quantized_model, tokenizer