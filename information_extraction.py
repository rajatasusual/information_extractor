import spacy
import logging
from code.rel_extractor import extract_relations
from code.spanbert import SpanBERT

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def process_coreference(coref, text):
    corefdoc = coref(text)
    resolved_text = ""
    for token in corefdoc:
        repres = corefdoc._.coref_chains.resolve(token)
        if repres:
            resolved_text += " " + " and ".join([t.text for t in repres])
        else:
            resolved_text += " " + token.text
    return resolved_text

def process_entities(doc, logger):
    entities = {ent.text: ent.label_ for ent in doc.ents}
    logger.info(f"Entities: {entities}")

    namedEntities = {}
    for ent in doc.ents:
        if ent.label_ not in namedEntities:
            namedEntities[ent.label_] = set()
        namedEntities[ent.label_].add(ent.text)
    for label, texts in namedEntities.items():
        logger.info(f"{label}: {', '.join(texts)}")

def print_details(doc, logger):
    # Basic token information
    logger.debug("--- Token Information ---")
    for token in doc:
        logger.debug(f"Text: {token.text:<20} Lemma: {token.lemma_:<20} POS: {token.pos_:<10} Tag: {token.tag_:<10} Dep: {token.dep_:<10}")
    # Sentences
    logger.debug("\n--- Sentences ---")
    for sent in doc.sents:
        logger.debug(f"Sentence: {sent.text}")

def main():
    DETAILS = True
    logger = setup_logging()

    # Load models
    coref = spacy.load('en_core_web_md')
    coref.add_pipe('coreferee')
    nlp = spacy.load('en_core_web_md')
    spanbert = SpanBERT("./assets/pretrained_spanbert")

    # Input text
    text = "C++ was developed by Bjarne Stroustrup at Bell Labs in Murray Hill, New Jersey, USA, while he was working on his PhD thesis and exploring ways to enhance the C programming language."

    # Coreference Resolution
    logger.info("--- Coreference Resolution ---")
    resolved_text = process_coreference(coref, text)
    logger.info(f"Resolved text: {resolved_text}")

    # Process with spaCy
    doc = nlp(resolved_text)

    if DETAILS:
        print_details(doc, logger)

    # Named Entity Recognition
    logger.info("--- Named Entities ---")
    process_entities(doc, logger)

    # Extract relations
    logger.info("--- Extracted Relations ---")
    relations = extract_relations(doc, spanbert)
    logger.info(relations)

    return relations, doc, resolved_text

if __name__ == "__main__":
    main()