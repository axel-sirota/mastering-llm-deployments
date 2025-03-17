"""
download_datasets.py

This script downloads dataset files from the Hugging Face Hub using the dataset repository IDs.
Datasets are saved into the 'local_datasets' folder.
"""
import os
from huggingface_hub import hf_hub_download, list_repo_files

def download_dataset_files(repo_id: str, target_dir: str, token: str = None):
    os.makedirs(target_dir, exist_ok=True)
    files = list_repo_files(repo_id, token=token, repo_type="dataset")
    print(f"Found {len(files)} files in dataset repo '{repo_id}'.")
    for file in files:
        print(f"Downloading file: {file}")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                token=token,
                repo_type="dataset"
            )
        except Exception as e:
            print(f"Error downloading file {file}: {e}")
    print(f"Completed downloading dataset '{repo_id}' to {target_dir}.")

if __name__ == "__main__":
    token = os.getenv("HF_TOKEN", None)
    datasets_to_download = {
        "imdb": "local_datasets/imdb",
        "dialogsum": "local_datasets/dialogsum"
    }
    for repo_id, target_dir in datasets_to_download.items():
        print(f"Downloading dataset: {repo_id}")
        download_dataset_files(repo_id, target_dir, token=token)
