# information_extractor  

## Overview  
[![CI](https://github.com/rajatasusual/information_extractor/actions/workflows/ci.yml/badge.svg)](https://github.com/rajatasusual/information_extractor/actions/workflows/ci.yml)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)  
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  

**information_extractor** is a Python package that combines **spaCy**, **coreferee**, and **SpanBERT** to extract structured relationships between entities in natural language text. It's purpose-built for anyone who wants to bridge NER, coreference resolution, and relation extraction into one streamlined pipeline.

## Features

### ✅ Entity Linking & Coreference Resolution
- Uses `spaCy` with `coreferee` to resolve pronouns and link entity mentions.
- Flexible support for multiple entity types: `PERSON`, `ORG`, `LOC`, `DATE`, etc.

### ✅ Relation Extraction with SpanBERT
- Uses fine-tuned SpanBERT model trained on TACRED.
- Handles subject/object marking and context-aware classification.
- Confidence scoring and de-duplication of extracted relations.
- GPU acceleration supported out of the box.

### ✅ CLI Interface
```bash
ie --text "Barack Obama was born in Hawaii." [--deps]
```
- `--deps`: Downloads and installs required pretrained models if not present.

## Installation

```bash
pip install information_extractor
```

### Optional: Download model dependencies
Run the following once to download SpanBERT, spaCy model, coreferee model:
```bash
ie --deps
```

Alternatively, you can import and run the dependency script directly:
```python
from information_extractor.dependency import setup_dependencies
setup_dependencies()
```

## Example Usage

```python
from information_extractor.pipeline import RelationExtractor

text = "Sundar Pichai is the CEO of Google. He lives in California."

extractor = RelationExtractor()
results = extractor.extract(text)

for relation in results:
    print(relation)
```

### Sample Output
```json
[
  {
    "subject": "Sundar Pichai",
    "object": "Google",
    "relation": "per:employee_of",
    "confidence": 0.92
  },
  ...
]
```

## Project Structure
```
information_extractor/
├── assets/
│   └── pretrained_spanbert/
├── dependency.py         # Downloads all model dependencies
├── pipeline.py           # Core logic for NLP + SpanBERT
├── main.py               # CLI entrypoint
```

## Pretrained Assets
Models are downloaded from hosted GitHub release assets:
- ✅ `SpanBERT` weights & config
- ✅ `en_core_web_md` spaCy model
- ✅ `coreferee_model_en` for coreference resolution
- ✅ `torch` wheel for reproducibility

## Citation  

This project builds on the work of Facebook Research. If you use **SpanBERT**, please cite:

```
@article{joshi2019spanbert,
  title={{SpanBERT}: Improving Pre-training by Representing and Predicting Spans},
  author={Mandar Joshi and Danqi Chen and Yinhan Liu and Daniel S. Weld and Luke Zettlemoyer and Omer Levy},
  journal={arXiv preprint arXiv:1907.10529},
  year={2019}
}
```

## License

MIT. See [LICENSE](./LICENSE) for full terms.  
Note: This project redistributes pretrained model weights for convenience under fair use for research.