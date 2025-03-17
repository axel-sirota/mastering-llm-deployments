"""
Lab 1a: Intro to Transfer Learning with Hugging Face

This script demonstrates how to load a pre-trained model and tokenizer from Hugging Face.
"""
from transformers import AutoTokenizer, AutoModel

def main():
    model_name = "local_models/flan-t5-xl"
    print(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # if fails, try adding `use_fast=False` and local_files_only=True
    print("Tokenizer loaded successfully.")
    
    print(f"Loading model {model_name}")
    model = AutoModel.from_pretrained(model_name)  # if fails, try adding `use_fast=False` and local_files_only=True
    print("Model loaded successfully.")

if __name__ == "__main__":
    main()
