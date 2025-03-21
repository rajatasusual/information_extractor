# spacy_coref_rel  

## Overview  

**spacy_coref_rel** is a tool that leverages **spaCy** for coreference resolution and **SpanBERT** for relation extraction. This project integrates named entity recognition (NER) with relation extraction to identify and analyze relationships between entities in text.  

## Installation  

To install and set up the project, run the following commands:  

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/rajatasusual/spacy_coref_rel.git
cd spacy_coref_rel
pip3 install -r requirements.txt
git lfs pull --include "assets/pretrained_spanbert/pytorch_model.bin"
```

Ensure that you have **Git LFS** installed to handle large model files.  

## Usage  

To extract relations using **spaCy** and **SpanBERT**, you can run the provided example script:  

```bash
python example_relations.py
```

### Example (Inside `example_relations.py`)  

```python
import spacy
from spanbert_module import SpanBERT  # Import SpanBERT model

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

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
