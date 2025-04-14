import spacy
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Improved spaCy to TACRED-compatible entity type mapping
spacy2bert: Dict[str, str] = {
    "ORG": "ORGANIZATION",
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "NORP": "ORGANIZATION",
    "FAC": "LOCATION",
    "TIME": "DATE",
    "DATE": "DATE",
}

# Extract minimal dependency-based context between two entities
def extract_relevant_tokens(sentence: spacy.tokens.Span, e1: spacy.tokens.Span, e2: spacy.tokens.Span) -> List[str]:
    start = min(e1.start, e2.start)
    end = max(e1.end, e2.end)
    subtree_tokens = set()

    for token in sentence[start:end]:
        if e1.root in token.ancestors or e2.root in token.ancestors or token in e1.subtree or token in e2.subtree:
            subtree_tokens.add(token)

    for token in e1:
        subtree_tokens.add(token)
    for token in e2:
        subtree_tokens.add(token)

    sorted_tokens = sorted(subtree_tokens, key=lambda t: t.i)
    return [t.text for t in sorted_tokens]

# Generate valid entity pairs from a sentence
def create_entity_pairs(sentence: spacy.tokens.Span) -> List[Tuple[List[str], Tuple[str, str, Tuple[int, int]], Tuple[str, str, Tuple[int, int]]]]:
    ents = sentence.ents
    entity_pairs = []

    for i, e1 in enumerate(ents):
        for j in range(i + 1, len(ents)):
            e2 = ents[j]

            if e1.text.lower() == e2.text.lower():
                continue
            if spacy2bert.get(e1.label_, "O") == "O" or spacy2bert.get(e2.label_, "O") == "O":
                continue

            e1_info = (e1.text, spacy2bert[e1.label_], (e1.start - sentence.start, e1.end - sentence.start - 1))
            e2_info = (e2.text, spacy2bert[e2.label_], (e2.start - sentence.start, e2.end - sentence.start - 1))
            tokens = extract_relevant_tokens(sentence, e1, e2)

            entity_pairs.append((tokens, e1_info, e2_info))

    return entity_pairs

# Relationship extraction from a resolved document
def extract_relations(doc: spacy.tokens.Doc, spanbert: Any, conf: float = 0.65) -> Dict[Tuple[str, str, str], float]:
    sentences = list(doc.sents)
    logger.info("Total number of sentences: %d", len(sentences))
    extracted_relations = defaultdict(float)

    for sentence in sentences:
        entity_pairs = create_entity_pairs(sentence)
        examples = []

        for tokens, e1_info, e2_info in entity_pairs:
            examples.append({"tokens": tokens, "subj": e1_info, "obj": e2_info})
            examples.append({"tokens": tokens, "subj": e2_info, "obj": e1_info})

        preds = spanbert.predict(examples)
        if preds is None:
            continue

        for ex, pred in zip(examples, preds):
            relation, confidence = pred[0], pred[1]
            if relation == 'no_relation':
                continue

            subj_text = ex["subj"][0]
            obj_text = ex["obj"][0]
            if confidence > conf:
                key = (subj_text, relation, obj_text)
                if confidence > extracted_relations[key]:
                    extracted_relations[key] = confidence

    return extracted_relations
