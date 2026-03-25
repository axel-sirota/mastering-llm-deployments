# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a course repository for "Mastering LLM Deployments" — 12 progressive labs covering LLM fine-tuning, compression, and deployment. It has a dual structure:
- `LLM_Deployments_Course/` — reference implementations
- `exercises/` — student-facing versions with placeholder sections

## Environment Setup

All Python commands must use the virtual environment:
```bash
.venv/bin/python3 script.py
.venv/bin/python3 -m pip install -r requirements.txt
```

Pre-download models and datasets before running labs (only needed once):
```bash
.venv/bin/python3 download_models.py
.venv/bin/python3 download_datasets.py
```

## Running Labs

**Training/fine-tuning labs (Labs 1–6):**
```bash
.venv/bin/python3 LLM_Deployments_Course/Lab<N>_<Name>/starter_code.py
```

**Service/inference labs:**
```bash
.venv/bin/python3 LLM_Deployments_Course/Lab7_Benchmarking/benchmark.py
.venv/bin/python3 LLM_Deployments_Course/Lab8_Chatbot/chatbot.py
.venv/bin/python3 LLM_Deployments_Course/Lab9_Gradio/app.py
```

**Docker (Lab 9, 11):**
```bash
docker build -t app .
docker run -p 7860:7860 app
```

**Terraform (Labs 10, 12):**
```bash
terraform init
terraform plan -out terraform_plans/<datetime>.tfplan
terraform apply terraform_plans/<datetime>.tfplan
```

## Architecture & Lab Progression

| Labs | Topic | Key Techniques |
|------|-------|---------------|
| 1a, 1b | Transfer Learning & Fine-Tuning | HuggingFace Trainer API |
| 2 | Sentiment (LoRA) | PEFT, Low-Rank Adaptation |
| 3 | Summarization | DialogSum dataset, Seq2Seq |
| 4 | Knowledge Distillation | Teacher/student, KL divergence, temperature scaling |
| 5 | Pruning | Structured/unstructured pruning |
| 6 | Quantization | QAT (qat.py), int8 |
| 7 | Benchmarking | BLEU/ROUGE, latency measurement |
| 8 | Chatbot | Single-turn summarization interface |
| 9 | Gradio UI | Web UI + Docker |
| 10 | AWS ECS + Terraform | ECS cluster, ECR, single task |
| 11 | HuggingFace Spaces | Dockerfile for HF Spaces |
| 12 | ECS Scaling + ALB | ALB, target groups, 2 desired tasks |

## Critical Patterns

**Model loading** — all models load from `local_models/` with `local_files_only=True`. Never download at runtime:
```python
model = AutoModel.from_pretrained("local_models/flan-t5-xl", local_files_only=True)
```

**Key models used:** `flan-t5-xl` (teacher), `flan-t5-large`, `t5-small` (student), `distilbert-base-uncased`

**Datasets:** `local_datasets/dialogsum` (summarization), `local_datasets/imdb` (sentiment)

**Port convention:** All services expose port `7860`

**CUDA pattern:** Labs detect CUDA and fall back to CPU:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Terraform Plans

Store all Terraform plans in `terraform_plans/` (gitignored) inside each Terraform lab directory. Always use `-out` with a datetime filename. Never run `terraform destroy` — user does that manually.

## Exercise Generation

`exercises.py` and `files.py` are course-authoring utilities that generate the `exercises/` directory from reference implementations. Run with `.venv/bin/python3 exercises.py` to regenerate exercises after changing reference implementations.
