"""
Lab 11: Deploying to Hugging Face Spaces

This script creates a summarisation interface for deployment on Hugging Face Spaces using Gradio.
"""
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model():
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(fn=summarize_text, inputs="text", outputs="text", title="HF Space Summarisation Chatbot",
                     description="Enter text and get a summarised version.")

if __name__ == "__main__":
    iface.launch()
