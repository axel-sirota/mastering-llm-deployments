"""
Lab 6: Model Quantization

This script demonstrates three quantization approaches for t5-small:
1. Training-Aware Quantization (simulation)
2. bfloat16 Quantization
3. int8 Dynamic Quantization

Usage: python starter_code.py [taq|bfloat16|int8]
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def quantize_model(model, mode):
    if mode == "bfloat16":
        pass  # YOUR CODE: cast the model to bfloat16 precision and print a confirmation
    elif mode == "int8":
        pass  # YOUR CODE: apply dynamic quantization to Linear layers using int8 dtype (use torch.ao.quantization) and print a confirmation
    elif mode == "taq":
        pass  # YOUR CODE: print a message indicating TAQ simulation was selected
    else:
        print("Invalid mode; returning original model.")
    return model

def main():
    if len(sys.argv) < 2:
        print("Usage: python starter_code.py [taq|bfloat16|int8]")
        sys.exit(1)
    mode = sys.argv[1]
    model_name = "local_models/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    print(f"Applying quantization mode: {mode}")
    quantized_model = quantize_model(model, mode)
    output_dir = f"./quantized_model_{mode}"
    quantized_model.save_pretrained(output_dir)
    print(f"Quantized model saved to {output_dir}")

if __name__ == "__main__":
    main()
