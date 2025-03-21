import spacy
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Any

# Configure logging for debug output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping between spaCy and spanBERT entity labels.
spacy2bert: Dict[str, str] = {
    "ORG": "ORGANIZATION",
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE",
    "NORP": "LOCATION",
    "FAC": "LOCATION",
    "PRODUCT": "LOCATION",
    "EVENT": "LOCATION",
    "WORK_OF_ART": "LOCATION",
    "LAW": "LOCATION",
    "LANGUAGE": "LOCATION",
    "ORDINAL": "LOCATION",
    "QUANTITY": "LOCATION",
    "TIME": "DATE",
    "MONEY": "LOCATION",
    "PERCENT": "LOCATION",
    "CARDINAL": "LOCATION",
}

bert2spacy: Dict[str, str] = {
    "ORGANIZATION": "ORG",
    "PERSON": "PERSON",
    "LOCATION": "LOC",
    "CITY": "GPE",
    "COUNTRY": "GPE",
    "STATE_OR_PROVINCE": "GPE",
    "DATE": "DATE",
}

def create_entity_pairs(sentence: spacy.tokens.Span) -> List[Tuple[List[str], Tuple[str, str, Tuple[int, int]], Tuple[str, str, Tuple[int, int]]]]:
    """
    Given a spaCy sentence, returns a list of entity pairs along with the tokenized text.
    Each pair is in the form: (tokens, e1_info, e2_info) where:
      - tokens: list of token texts for the entire sentence.
      - e1_info: tuple (entity_text, mapped_label, (start_idx, end_idx)) for the subject.
      - e2_info: similar tuple for the object.
    """
    ents = sentence.ents
    entity_pairs = []
    # Use the full sentence as context
    tokens = [token.text for token in sentence]
    for i, e1 in enumerate(ents):
        for j in range(i + 1, len(ents)):
            e2 = ents[j]
            if e1.text.lower() == e2.text.lower():
                continue
            # Compute indices relative to the sentence.
            # Note: spaCy's e.end is exclusive, so subtract 1 to get an inclusive index.
            e1_info = (e1.text, spacy2bert.get(e1.label_, e1.label_), (e1.start - sentence.start, e1.end - sentence.start - 1))
            e2_info = (e2.text, spacy2bert.get(e2.label_, e2.label_), (e2.start - sentence.start, e2.end - sentence.start - 1))
            entity_pairs.append((tokens, e1_info, e2_info))
    return entity_pairs

def extract_relations(doc: spacy.tokens.Doc, 
                      spanbert: Any,
                      conf: float = 0.65
                     ) -> Dict[Tuple[str, str, str], float]:
    """
    Processes the spaCy document and extracts entity relationships using spanBERT.
    
    For each sentence in the document:
      - It creates candidate entity pairs.
      - Generates examples (including swapping subject/object).
      - Uses spanBERT to predict relations.
      - Returns a dictionary mapping (subject, relation, object) to the highest confidence observed.
    """
    sentences = list(doc.sents)
    logger.info("Total number of sentences: %d", len(sentences))
    extracted_relations = defaultdict(float)

    for sentence in sentences:
        logger.debug("Processing sentence: %s", sentence.text)
        entity_pairs = create_entity_pairs(sentence)
        examples = []
        for tokens, e1_info, e2_info in entity_pairs:
            # Add both subject-object orders to account for directionality.
            examples.append({"tokens": tokens, "subj": e1_info, "obj": e2_info})
            examples.append({"tokens": tokens, "subj": e2_info, "obj": e1_info})

        preds = spanbert.predict(examples)
        if preds is None:
            logger.debug("No predictions returned by spanBERT.")
            continue

        for ex, pred in zip(examples, preds):
            relation, confidence = pred[0], pred[1]
            if relation == 'no_relation':
                continue

            subj_text = ex["subj"][0]
            obj_text = ex["obj"][0]
            logger.debug("Extracted relation: %s (Confidence: %.3f) between '%s' and '%s'", 
                         relation, confidence, subj_text, obj_text)

            if confidence > conf:
                key = (subj_text, relation, obj_text)
                if confidence > extracted_relations[key]:
                    extracted_relations[key] = confidence
                    logger.debug("Relation added/updated: %s", key)
                else:
                    logger.debug("Duplicate relation with lower confidence; skipping.")
            else:
                logger.debug("Confidence below threshold; skipping relation.")

    return extracted_relations
