"""
Lab 1a: Intro to Transfer Learning with Hugging Face

This script demonstrates how to load a pre-trained model and tokenizer from local files.
- Models are loaded using local_files_only=True.
- The tokenizer is loaded with use_fast=False.
"""
from transformers import AutoTokenizer, AutoModel

def main():
    model_name = "local_models/flan-t5-xl"
    print(f"Loading tokenizer for {model_name} with use_fast=False")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, local_files_only=True)
    print("Tokenizer loaded successfully from local files.")
    
    print(f"Loading model {model_name} with local_files_only=True")
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    print("Model loaded successfully from local files.")

if __name__ == "__main__":
    main()
