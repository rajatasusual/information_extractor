# information_extractor  

## Overview  
[![CI](https://github.com/rajatasusual/information_extractor/actions/workflows/ci.yml/badge.svg)](https://github.com/rajatasusual/information_extractor/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**information_extractor** is a tool that leverages **spaCy** for coreference resolution and **SpanBERT** for relation extraction. This project integrates named entity recognition (NER) with relation extraction to identify and analyze relationships between entities in text.  

## Features

### SpanBERT Model
- Pre-trained model for relation extraction between entities
- Supports multiple entity types (PERSON, ORGANIZATION, LOCATION, etc.)
- Handles special token markers for subject and object entities
- Uses BERT architecture for sequence classification
- GPU acceleration support when available
- Configurable batch size and sequence length

### Entity Processing
- Maps between spaCy and SpanBERT entity labels
- Supports common entity types:
    - Organizations (ORG)
    - Persons (PERSON)
    - Locations (GPE, LOC)
    - Dates (DATE)
    - And more

### Relation Extraction
- Creates entity pairs from spaCy sentences
- Handles bidirectional relationships
- Configurable confidence threshold
- Deduplicates relations with confidence scoring
- Returns structured relation tuples
- Detailed logging for debugging

### Pretrained Models

The `assets` directory contains the following pretrained models:

- **pretrained_spanbert/** finetuned for TARCED use cases.
- **corefereee_model_en** from stanford research
- **en_core_web_md-3.50** from spaCy

## Installation  

To install and set up the project, run the following commands:  

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/rajatasusual/information_extractor.git
cd information_extractor
pip3 install -r requirements.txt
git lfs pull --include "assets/pretrained_spanbert/pytorch_model.bin"
```

Ensure that you have **Git LFS** installed to handle large model files.  

## Usage  

To extract relations using **spaCy** and **SpanBERT**, you can run the provided example script:  

```bash
python main.py
```

### Example (Inside `main.py`)  

```python
import spacy
from spanbert_module import SpanBERT  # Import SpanBERT model

# Load spaCy NLP model
nlp = spacy.load("en_core_web_md")

# Sample text
text = "Bill Gates founded Microsoft. Microsoft is headquartered in Redmond."

# Process text with spaCy
doc = nlp(text)

# Load SpanBERT
pretrained_dir = "assets/pretrained_spanbert"
spanbert = SpanBERT(pretrained_dir=pretrained_dir)

# Extract relations
relations = spanbert.extract_relations(doc)
print(relations)
```

## Acknowledgments  

This project integrates **SpanBERT** from **Facebook Research**. If you use this project, please cite:  

```
@article{joshi2019spanbert,
    title={{SpanBERT}: Improving Pre-training by Representing and Predicting Spans},
    author={Mandar Joshi and Danqi Chen and Yinhan Liu and Daniel S. Weld and Luke Zettlemoyer and Omer Levy},
    journal={arXiv preprint arXiv:1907.10529},
    year={2019}
}
```

## License & Disclaimer  

This project is intended for research and educational purposes. The SpanBERT model belongs to **Facebook Research**, and its use must comply with their licensing terms. We are not affiliated with Facebook Research.  
