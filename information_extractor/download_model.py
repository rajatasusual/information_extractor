import os
import urllib.request
import shutil

def download_spanbert():
    model_dir = os.path.join(os.path.dirname(__file__), "assets", "pretrained_spanbert")
    os.makedirs(model_dir, exist_ok=True)

    model_url = "https://github.com/rajatasusual/information_extractor/raw/refs/heads/master/assets/pretrained_spanbert/pytorch_model.bin"
    EXPECTED_SIZE = 667334021  # bytes (667.3 MB)

    target_path = os.path.join(model_dir, "pytorch_model.bin")
    config_src = os.path.join(os.path.dirname(__file__), "assets", "config.json")
    config_dst = os.path.join(model_dir, "config.json")

    if not os.path.exists(target_path) or os.path.getsize(target_path) != EXPECTED_SIZE:
        print(f"Model missing or corrupted. Downloading SpanBERT model to {target_path}...")
        urllib.request.urlretrieve(model_url, target_path)
        print("Download complete.")
    else:
        print(f"Model already exists at {target_path}. Skipping download.")

    if not os.path.exists(config_dst):
        shutil.copy(config_src, config_dst)
        print("Copied config.json.")
    else:
        print("config.json already exists. Skipping.")
