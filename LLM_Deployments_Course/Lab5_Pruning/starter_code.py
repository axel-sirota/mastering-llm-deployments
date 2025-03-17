"""
Lab 5: Model Pruning

This script demonstrates how to prune a pre-trained model to remove less important weights.
We use PyTorch's pruning utilities.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.nn.utils.prune as prune

def main():
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    print("Before pruning:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=0.2)
            prune.remove(module, 'weight')
    print("After pruning:")
    total_params_pruned = sum(p.numel() for p in model.parameters())
    print(f"Total parameters (count remains the same, but many weights are zeroed): {total_params_pruned}")
    model.save_pretrained("./pruned_model")
    print("Pruned model saved to ./pruned_model")

if __name__ == "__main__":
    main()
