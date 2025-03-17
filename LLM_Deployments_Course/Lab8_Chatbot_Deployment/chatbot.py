"""
Lab 8: Chatbot Deployment

This script implements a simple, stateless summarisation chatbot.
User input is processed to generate a summary.
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model():
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    return tokenizer, model

def summarize_text(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    outputs = model.generate(**inputs)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
