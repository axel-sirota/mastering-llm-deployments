"""
Lab 8: Chatbot Deployment

This script implements a simple, stateless summarisation chatbot.
User input is processed to generate a summary.
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model():
    model_name = None  # YOUR CODE: path to local t5-small model directory
    tokenizer = None  # YOUR CODE: load tokenizer from model_name with local_files_only=True
    model = None  # YOUR CODE: load the seq2seq model from model_name with local_files_only=True
    return tokenizer, model

def summarize_text(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    outputs = None  # YOUR CODE: run model.generate() with the tokenized inputs
    summary = None  # YOUR CODE: decode the first output sequence, skipping special tokens
    return summary

def main():
    tokenizer, model = load_model()
    print("Welcome to the Summarisation Chatbot!")
    print("Type your conversation text (or 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        summary = summarize_text(tokenizer, model, user_input)
        print("Chatbot Summary:", summary)

if __name__ == "__main__":
    main()
