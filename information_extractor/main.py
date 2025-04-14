import logging
import os
import argparse

from information_extractor.dependencies import setup_dependencies

# Global variables for models
SPACY_MODEL = None
SPANBERT_MODEL = None

def main():
    parser = argparse.ArgumentParser(description="Run Information Extractor")
    parser.add_argument('--deps', action='store_true', help='Download SpanBERT and required model dependencies')
    args = parser.parse_args()

    if args.deps:
        setup_dependencies()
        return
    
    import spacy

    from information_extractor.iecode.rel_extractor import extract_relations
    from information_extractor.iecode.spanbert import SpanBERT

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

        resolved_text = process_coreference(text)
        logger.info(f"Resolved text: {resolved_text}")
        doc = spacy_nlp(resolved_text, logger)

        if details:
            print_details(doc, logger)

        named_entities = process_entities(doc, logger)

        spanbert = load_spanbert_model()
        relations = extract_relations(doc, spanbert)
        logger.info(relations)

        return relations, named_entities

    input_text = """
The history of Rome spans over two and a half thousand years, from the legendary founding of the city of Rome in 753 BC to the fall of the Western Roman Empire in 476 AD. The city, which was founded by Romulus and Remus, was originally a small settlement on the Palatine Hill.
The Roman Empire was a vast and powerful state that stretched from the British Isles to Egypt and from Spain to Syria. It was a major centre of trade, commerce and culture, and was home to many famous landmarks such as the Colosseum, the Pantheon and the Forum Romanum. The empire was ruled by a series of emperors, including Augustus, Trajan, Marcus Aurelius and Constantine. It was also the centre of Christianity, with the Pope based in Rome.
"""
    extract_information(input_text)