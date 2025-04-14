import os
import urllib.request
import shutil
from tqdm import tqdm

def download_spanbert():
    model_dir = os.path.join(os.path.dirname(__file__), "assets", "pretrained_spanbert")
    os.makedirs(model_dir, exist_ok=True)

    model_url = "https://github.com/rajatasusual/information_extractor/raw/refs/heads/master/assets/pretrained_spanbert/pytorch_model.bin"
    EXPECTED_SIZE = 667334021  # bytes (667.3 MB)
    target_path = os.path.join(model_dir, "pytorch_model.bin")
    
    config_url = "https://github.com/rajatasusual/information_extractor/raw/refs/heads/master/assets/pretrained_spanbert/config.json"
    
    if not os.path.exists(target_path) or os.path.getsize(target_path) != EXPECTED_SIZE:
        print(f"Model missing or corrupted. Downloading SpanBERT model to {target_path}...")
        with urllib.request.urlopen(model_url) as response:
            with open(target_path, "wb") as fh:
                with tqdm(total=EXPECTED_SIZE, unit="B", unit_scale=True, desc="Downloading model") as pbar:
                    while True:
                        data = response.read(4096)
                        if not data:
                            break
                        fh.write(data)
                        pbar.update(len(data))
        with urllib.request.urlopen(config_url) as response:
            print("Downloading config.json...")
            with open(os.path.join(model_dir, "config.json"), "wb") as fh:
                fh.write(response.read())
        print("Download complete.")
    else:
        print(f"Model already exists at {target_path}. Skipping download.")