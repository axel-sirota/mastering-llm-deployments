import os
from datasets import load_dataset

def download_dataset(dataset_name: str, split: str = None, target_dir: str = None):
    """
    Downloads a dataset using Hugging Face Datasets and saves it to disk.
    
    Args:
        dataset_name (str): The name of the dataset to download (e.g., "imdb" or "dialogsum").
        split (str, optional): The split to load (e.g., "train"). If None, loads the whole dataset.
        target_dir (str, optional): The directory where the dataset will be saved.
                                      Defaults to the dataset name.
    """
    if target_dir is None:
        target_dir = dataset_name
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Loading dataset '{dataset_name}'" + (f" (split='{split}')" if split else ""))
    try:
        if split:
            dataset = load_dataset(dataset_name, split=split)
        else:
            dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return
    
    print(f"Saving dataset '{dataset_name}' to {target_dir}...")
    dataset.save_to_disk(target_dir)
    print(f"Dataset '{dataset_name}' saved successfully to {target_dir}.")

if __name__ == "__main__":
    datasets_to_download = {
        "imdb": {"split": "train", "target_dir": "local_datasets/imdb"},
        "knkarthick/dialogsum": {"split": None, "target_dir": "local_datasets/dialogsum"}
    }
    
    for ds_name, config in datasets_to_download.items():
        print(f"Downloading dataset: {ds_name}")
        download_dataset(dataset_name=ds_name, split=config.get("split"), target_dir=config.get("target_dir"))
