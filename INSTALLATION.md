# 🛠 Installation Guide for `information-extractor`

Whether you're a user just looking to run the CLI or a developer planning to contribute — here's how to set up everything cleanly.

---

## ✅ For Users (via PyPI)

This is the **recommended** method if you're just using the tool.

### 1. Install the package
```bash
pip install information-extractor
```

### 2. Download required models and dependencies
The core models (SpanBERT, spaCy model, Coreferee, Torch) are large and managed separately.

Run:
```bash
ie --deps
```

This will:
- Download the SpanBERT model and config
- Install the required spaCy model
- Install Torch and coreference model

You're done! Use the CLI:
```bash
ie --input "Alice works at Google."
```

---

## 🧪 For Developers (from Source)

Follow this if you're modifying the code or contributing.

### 1. Clone the repository **without auto-downloading LFS files**
This prevents huge binaries from being fetched immediately.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/rajatasusual/information_extractor.git
cd information_extractor
```

### 2. Set up the environment

**Option A: Editable install (preferred for development)**
```bash
pip install -e .
```

**Option B: Traditional requirements-based install**
```bash
pip install -r requirements.txt
```

### 3. Pull the model files (optional if using `--deps`)
If you want to use Git LFS instead of downloading via CLI:

```bash
git lfs install
git lfs pull --include "assets/pretrained_spanbert/pytorch_model.bin"
```

Otherwise, just run:
```bash
ie --deps
```

---

## 🧩 Notes
- Make sure `python --version` is 3.7 or above (>=3.10 recommended).
- Git LFS is only needed if pulling models manually via Git.
- `ie --deps` handles:
  - ✅ SpanBERT model & config
  - ✅ en-core-web-md spaCy model
  - ✅ Coreferee English model
  - ✅ PyTorch (from hosted `.whl`)
