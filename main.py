import spacy
import logging
import os
from iecode.rel_extractor import extract_relations
from iecode.spanbert import SpanBERT

# Global variables for models
SPACY_MODEL = None
SPANBERT_MODEL = None

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_spacy_model():
    global SPACY_MODEL
    if SPACY_MODEL is None:
        SPACY_MODEL = spacy.load('en_core_web_md')
    return SPACY_MODEL

def load_spanbert_model():
    global SPANBERT_MODEL
    if SPANBERT_MODEL is None:
        SPANBERT_MODEL = SpanBERT(os.path.join(os.path.dirname(__file__), "assets/pretrained_spanbert"))
    return SPANBERT_MODEL

def process_coreference(text):
    nlp = load_spacy_model()
    if 'coreferee' not in nlp.pipe_names:
        nlp.add_pipe('coreferee', last=True)

    corefdoc = nlp(text)
    resolved_text = []
    for token in corefdoc:
        repres = corefdoc._.coref_chains.resolve(token)
        if repres:
            resolved_text.append(" and ".join([t.text for t in repres]))
        else:
            resolved_text.append(token.text)
    return " ".join(resolved_text)

def spacy_nlp(text, logger):
    nlp = load_spacy_model()
    logger.info("--- Named Entity Recognition ---")
    return nlp(text)

def process_entities(doc, logger):
    named_entities = {}
    for ent in doc.ents:
        named_entities.setdefault(ent.label_, set()).add(ent.text)

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
    doc = spacy_nlp(resolved_text, logger)

    if details:
        print_details(doc, logger)

    logger.info("--- Named Entities ---")
    named_entities = process_entities(doc, logger)

    logger.info("--- Extracted Relations ---")
    spanbert = load_spanbert_model()
    relations = extract_relations(doc, spanbert)
    logger.info(relations)

    return relations, named_entities

if __name__ == "__main__":
    input_text = "C++ was developed by Bjarne Stroustrup at Bell Labs in Murray Hill, New Jersey, USA, while he was working on his PhD thesis and exploring ways to enhance the C programming language."
    extract_information(input_text)
