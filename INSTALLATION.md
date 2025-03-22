# Installation Instructions

1. Clone the repository with Git LFS skip option:
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/rajatasusual/information_extractor.git
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

3. Pull the required model files:
```bash
git lfs pull --include "assets/pretrained_spanbert/pytorch_model.bin"
```

**Note**: Make sure you have Git LFS installed on your system before running these commands.
