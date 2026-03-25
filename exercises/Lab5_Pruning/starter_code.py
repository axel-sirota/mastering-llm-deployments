"""
Lab 5: Model Pruning

This script demonstrates how to prune a model (t5-small) to reduce its effective complexity.
We use PyTorch's pruning utilities to remove 20% of weights in Linear layers.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.nn.utils.prune as prune

def main():
    model_name = "local_models/t5-small"
    tokenizer = None  # YOUR CODE: load the tokenizer from model_name with local_files_only=True
    model = None  # YOUR CODE: load the seq2seq model from model_name with local_files_only=True
    print("Before pruning:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            pass  # YOUR CODE: apply l1_unstructured pruning on the module weight (amount=0.2), then remove the reparameterization
    print("After pruning:")
    total_params_pruned = sum(p.numel() for p in model.parameters())
    print(f"Total parameters (count remains the same, but many weights are zeroed): {total_params_pruned}")
    model.save_pretrained("./pruned_model")
    print("Pruned model saved to ./pruned_model")

if __name__ == "__main__":
    main()
