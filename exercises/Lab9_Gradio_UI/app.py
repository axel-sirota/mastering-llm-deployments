"""
Lab 9: Gradio UI for Summarisation Chatbot

This script creates a web-based interface using Gradio for the summarisation chatbot.
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
    outputs = # [EXERCISE] Add any parameters you think are necessary.
    return # [EXERCISE] Add any parameters you think are necessary.

iface = gr.Interface(fn=summarize_text, inputs="text", outputs="text", title="Summarisation Chatbot",
                     description="Enter a conversation or text to get a summarised version.")

if __name__ == "__main__":
    iface.launch()
