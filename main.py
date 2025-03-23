import spacy
import logging
import os

from iecode.rel_extractor import extract_relations
from iecode.spanbert import SpanBERT

# Load models once at startup
print("Loading models...")  # Debugging message
coref = spacy.load('en_core_web_md')
coref.add_pipe('coreferee')
nlp = spacy.load('en_core_web_md')
spanbert = SpanBERT(os.path.join(os.path.dirname(__file__), "assets/pretrained_spanbert"))
print("Models loaded!")  # Debugging message

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def process_coreference(text):
    corefdoc = coref(text)
    resolved_text = ""
    for token in corefdoc:
        repres = corefdoc._.coref_chains.resolve(token)
        if repres:
            resolved_text += " " + " and ".join([t.text for t in repres])
        else:
            resolved_text += " " + token.text
    return resolved_text.strip()

def process_entities(doc, logger):
    entities = {ent.text: ent.label_ for ent in doc.ents}
    logger.info(f"Entities: {entities}")

    named_entities = {}
    for ent in doc.ents:
        if ent.label_ not in named_entities:
            named_entities[ent.label_] = set()
        named_entities[ent.label_].add(ent.text)

    for label, texts in named_entities.items():
        logger.info(f"{label}: {', '.join(texts)}")
    return named_entities

def print_details(doc, logger):
    logger.debug("--- Token Information ---")
    for token in doc:
        logger.debug(f"Text: {token.text:<20} Lemma: {token.lemma_:<20} POS: {token.pos_:<10} Tag: {token.tag_:<10} Dep: {token.dep_:<10}")

    logger.debug("\n--- Sentences ---")
    for sent in doc.sents:
        logger.debug(f"Sentence: {sent.text}")

def extract_information(text, details=True):
    logger = setup_logging()

    logger.info("--- Coreference Resolution ---")
    resolved_text = process_coreference(text)
    logger.info(f"Resolved text: {resolved_text}")

    doc = nlp(resolved_text)

    if details:
        print_details(doc, logger)

    logger.info("--- Named Entities ---")
    named_entities = process_entities(doc, logger)

    logger.info("--- Extracted Relations ---")
    relations = extract_relations(doc, spanbert)
    logger.info(relations)

    return relations, doc, resolved_text, named_entities

if __name__ == "__main__":
    input_text = "C++ was developed by Bjarne Stroustrup at Bell Labs in Murray Hill, New Jersey, USA, while he was working on his PhD thesis and exploring ways to enhance the C programming language."
    extract_information(input_text)
