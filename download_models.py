import os
from huggingface_hub import hf_hub_download, list_repo_files

# Skip Flax, TF, and ONNX weights — we only need PyTorch/safetensors
SKIP_PATTERNS = (
    "flax_model", "tf_model", ".msgpack", "onnx/",
    ".ot", "rust_model", "coreml/",
)

def should_skip(filename):
    return any(p in filename for p in SKIP_PATTERNS)

def download_model(repo_id: str, target_dir: str, token: str = None):
    os.makedirs(target_dir, exist_ok=True)
    files = list_repo_files(repo_id, token=token)
    to_download = [f for f in files if not should_skip(f)]
    skipped = len(files) - len(to_download)
    print(f"Found {len(files)} files in {repo_id}, downloading {len(to_download)} (skipping {skipped} non-PyTorch).")
    for file in to_download:
        dest = os.path.join(target_dir, file)
        if os.path.exists(dest):
            print(f"  Already exists: {file}, skipping.")
            continue
        print(f"  Downloading {file} ...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=target_dir,
                token=token
            )
        except Exception as e:
            print(f"  Error downloading {file}: {e}")
    print(f"Completed download for {repo_id} to {target_dir}.\n")

if __name__ == "__main__":
    token = os.getenv("HF_TOKEN", None)
    models = {
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english": "local_models/distilbert-base-uncased-finetuned-sst-2-english",
        "microsoft/Phi-3.5-mini-instruct": "local_models/Phi-3.5-mini-instruct",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "local_models/paraphrase-multilingual-MiniLM-L12-v2",
        "google/flan-t5-xl": "local_models/flan-t5-xl",
        "google/flan-t5-large": "local_models/flan-t5-large"
    }
    for repo_id, target_dir in models.items():
        print(f"=== Downloading model: {repo_id} ===")
        download_model(repo_id, target_dir, token=token)
