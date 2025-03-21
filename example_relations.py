from spacy_help_functions import extract_relations
import spacy
from spanbert import SpanBERT

# Array of raw texts
raw_texts = [
    "C++ was developed by Bjarne Stroustrup at Bell Labs in Murray Hill, New Jersey, USA, while he was working on his PhD thesis and exploring ways to enhance the C programming language.",
    "Bell Labs, which has been a pioneer in technological innovation since its founding, is now a research and development subsidiary of Nokia, continuing its legacy of breakthrough discoveries.",
    "Murray Hill, a peaceful suburban city located in New Jersey, has become famous for hosting the renowned Bell Labs facility where numerous technological breakthroughs occurred.",
    "Nokia, headquartered in Finland, is a multinational corporation that specializes in telecommunications, information technology, and consumer electronics, while maintaining a strong presence in research through Bell Labs.",
    "Bjarne Stroustrup, who was born in Denmark and later moved to the United States, is a distinguished computer scientist known primarily for creating and developing the C++ programming language at Bell Labs during the 1980s."
]

entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]

# Load spacy model
nlp = spacy.load("en_core_web_md")  

# Load pre-trained SpanBERT model
spanbert = SpanBERT("./assets/pretrained_spanbert")  

# Process each text and extract relations
all_relations = {}
for i, text in enumerate(raw_texts):
    # Apply spacy model to raw text
    doc = nlp(text)
    
    # Extract relations
    relations = extract_relations(doc, spanbert)
    all_relations[f"Text {i+1}"] = dict(relations)

# Print all relations
for text_id, rels in all_relations.items():
    print(f"\n{text_id} Relations:")
    print(rels)
