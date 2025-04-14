import os
import urllib.request
import zipfile
import subprocess
from tqdm import tqdm


BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")


def download_with_progress(url, target_path, expected_size=None, desc="Downloading"):
    if os.path.exists(target_path) and (expected_size is None or os.path.getsize(target_path) == expected_size):
        print(f"{os.path.basename(target_path)} already exists. Skipping.")
        return

    print(f"{os.path.basename(target_path)} missing or corrupted. Downloading from {url}...")

    with urllib.request.urlopen(url) as response:
        with open(target_path, "wb") as fh:
            total = expected_size if expected_size else int(response.headers.get("content-length", 0))
            with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
                while True:
                    data = response.read(4096)
                    if not data:
                        break
                    fh.write(data)
                    pbar.update(len(data))
    print(f"Download complete: {target_path}")


def install_wheel(url):
    wheel_path = os.path.join(ASSETS_DIR, os.path.basename(url))
    download_with_progress(url, wheel_path, desc="Downloading wheel")
    print(f"Installing {os.path.basename(url)}...")
    subprocess.check_call(["pip", "install", wheel_path])


def extract_zip(url, extract_to):
    zip_path = os.path.join(ASSETS_DIR, "temp.zip")
    download_with_progress(url, zip_path, desc="Downloading ZIP")
    print(f"Extracting to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)


def download_spanbert():
    model_dir = os.path.join(ASSETS_DIR, "pretrained_spanbert")
    os.makedirs(model_dir, exist_ok=True)

    model_url = "https://github.com/rajatasusual/information_extractor/raw/refs/heads/master/assets/pretrained_spanbert/pytorch_model.bin"
    EXPECTED_SIZE = 667334021  # bytes
    target_path = os.path.join(model_dir, "pytorch_model.bin")
    
    config_url = "https://github.com/rajatasusual/information_extractor/raw/refs/heads/master/assets/pretrained_spanbert/config.json"
    config_path = os.path.join(model_dir, "config.json")

    download_with_progress(model_url, target_path, expected_size=EXPECTED_SIZE, desc="Downloading SpanBERT model")
    download_with_progress(config_url, config_path, desc="Downloading config.json")


def setup_dependencies():
    print("Setting up all dependencies...")

    download_spanbert()

    # Install torch wheel
    install_wheel("https://github.com/rajatasusual/information_extractor/raw/master/assets/torch-2.6.0-py3-none-any.whl")

    # Install spaCy model wheel
    install_wheel("https://github.com/rajatasusual/information_extractor/raw/master/assets/en_core_web_md-3.5.0-py3-none-any.whl")

    # Download and install coreferee model
    coreferee_model_dir = os.path.join(ASSETS_DIR, "coreferee_model_en")
    extract_zip(
        "https://github.com/rajatasusual/information_extractor/raw/master/assets/coreferee_model_en.zip",
        coreferee_model_dir
    )

    print("Installing coreferee-model-en...")
    subprocess.check_call(["pip", "install", coreferee_model_dir])

    print("✅ All dependencies are set up.")

