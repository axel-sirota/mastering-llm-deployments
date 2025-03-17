"""
download_models.py

This script downloads model files from the Hugging Face Hub using the provided repository IDs.
Models are saved into the 'local_models' folder.
"""
import os
from huggingface_hub import hf_hub_download, list_repo_files

def download_model(repo_id: str, target_dir: str, token: str = None):
    os.makedirs(target_dir, exist_ok=True)
    files = list_repo_files(repo_id, token=token, repo_type="model")
    print(f"Found {len(files)} files in model repo '{repo_id}'.")
    for file in files:
        print(f"Downloading file: {file}")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                token=token,
                repo_type="model"
            )
        except Exception as e:
            print(f"Error downloading file {file}: {e}")
    print(f"Completed downloading model '{repo_id}' to {target_dir}.")

if __name__ == "__main__":
    token = os.getenv("HF_TOKEN", None)
    models_to_download = {
        "google/flan-t5-xl": "local_models/flan-t5-xl",
        "distilbert-base-uncased-finetuned-sst-2-english": "local_models/distilbert-base-uncased-finetuned-sst-2-english"
    }
    for repo_id, target_dir in models_to_download.items():
        print(f"Downloading model: {repo_id}")
        download_model(repo_id, target_dir, token=token)
