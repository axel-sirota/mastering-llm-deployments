import os

folders = [
    "LLM_Deployments_Course",
    "LLM_Deployments_Course/Lab1a_Intro_Transfer_Learning",
    "LLM_Deployments_Course/Lab1b_Basic_FineTuning",
    "LLM_Deployments_Course/Lab2_Sentiment_Analysis_IMDB",
    "LLM_Deployments_Course/Lab3_Summarisation_DialogSum",
    "LLM_Deployments_Course/Lab4_Distillation_TrainingLoop",
    "LLM_Deployments_Course/Lab5_Pruning",
    "LLM_Deployments_Course/Lab6_Quantization",
    "LLM_Deployments_Course/Lab7_Benchmarking",
    "LLM_Deployments_Course/Lab8_Chatbot_Deployment",
    "LLM_Deployments_Course/Lab9_Gradio_UI",
    "LLM_Deployments_Course/Lab10_AWS_ECS_Terraform",
    "LLM_Deployments_Course/Lab11_HF_Space",
    "LLM_Deployments_Course/Lab12_AWS_ECS_Scaling_Terraform"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")
