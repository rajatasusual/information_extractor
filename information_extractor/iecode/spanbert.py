# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""
Run SpanBERT on general test examples
Scripts adopted from https://github.com/facebookresearch/SpanBERT
"""
import os
import psutil
import random
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from information_extractor.iecode.pytorch_pretrained_bert.modeling import BertForSequenceClassification
from information_extractor.iecode.pytorch_pretrained_bert.tokenization import BertTokenizer
from scipy.special import softmax

# Special tokens used by the model.
CLS = "[CLS]"
SEP = "[SEP]"

label_list = [
    'no_relation', 'per:title', 'org:top_members/employees', 'per:employee_of', 
    'org:alternate_names', 'org:country_of_headquarters', 'per:countries_of_residence', 
    'per:age', 'org:city_of_headquarters', 'per:cities_of_residence', 
    'per:stateorprovinces_of_residence', 'per:origin', 'org:subsidiaries', 
    'org:parents', 'per:spouse', 'org:stateorprovince_of_headquarters', 
    'per:children', 'per:other_family', 'org:members', 'per:siblings', 'per:parents', 
    'per:schools_attended', 'per:date_of_death', 'org:founded_by', 'org:member_of', 
    'per:cause_of_death', 'org:website', 'org:political/religious_affiliation', 
    'per:alternate_names', 'org:founded', 'per:city_of_death', 'org:shareholders', 
    'org:number_of_employees/members', 'per:charges', 'per:city_of_birth', 
    'per:date_of_birth', 'per:religion', 'per:stateorprovince_of_death', 
    'per:stateorprovince_of_birth', 'per:country_of_birth', 'org:dissolved', 
    'per:country_of_death'
]

# Special tokens: these can be customized or expanded.
special_tokens = {
    'SUBJ_START': '[unused1]',
    'SUBJ_END': '[unused2]',
    'OBJ_START': '[unused3]',
    'OBJ_END': '[unused4]',
    # Optionally, include type-specific markers:
    'SUBJ=PERSON': '[unused5]',
    'SUBJ=ORGANIZATION': '[unused6]',
    'OBJ=PERSON': '[unused7]',
    'OBJ=ORGANIZATION': '[unused8]',
    # Add more as needed...
}

class InputExample(object):
    """A single training/test example for span pair classification."""
    def __init__(self, sentence, span1, span2, ner1, ner2):
        self.sentence = sentence  # list of tokens (strings)
        self.span1 = span1        # tuple: (start_index, end_index) for subject
        self.span2 = span2        # tuple: (start_index, end_index) for object
        self.ner1 = ner1          # entity type for subject (e.g., "PERSON")
        self.ner2 = ner2          # entity type for object (e.g., "ORGANIZATION")

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

def convert_examples_to_features(examples, max_seq_length, tokenizer, special_tokens):
    """
    Converts a list of examples into InputFeatures for SpanBERT.
    Each example is expected to be a dict with keys:
      - 'tokens': list of token strings
      - 'subj': tuple (entity_text, entity_type, (start, end))
      - 'obj': tuple (entity_text, entity_type, (start, end))
    """
    def create_examples(dataset):
        """Create InputExample instances from dataset."""
        exs = []
        for example in dataset:
            exs.append(InputExample(
                sentence=example['tokens'],
                span1=example['subj'][2],
                span2=example['obj'][2],
                ner1=example['subj'][1],
                ner2=example['obj'][1]
            ))
        return exs

    examples = create_examples(examples)
    features = []
    for ex in examples:
        tokens = [CLS]
        subj_start, subj_end = ex.span1
        obj_start, obj_end = ex.span2

        for i, token in enumerate(ex.sentence):
            # If this token is the first token of a subject, insert a subject marker.
            if i == subj_start:
                subj_marker = special_tokens.get("SUBJ=%s" % ex.ner1, special_tokens["SUBJ_START"])
                tokens.append(subj_marker)
            # If this token is the first token of an object, insert an object marker.
            if i == obj_start:
                obj_marker = special_tokens.get("OBJ=%s" % ex.ner2, special_tokens["OBJ_START"])
                tokens.append(obj_marker)
            # Tokenize and append the current token.
            tokenized = tokenizer.tokenize(token)
            tokens.extend(tokenized)
            # If this token is the last token of a subject, add the subject end marker.
            if i == subj_end:
                tokens.append(special_tokens["SUBJ_END"])
            # If this token is the last token of an object, add the object end marker.
            if i == obj_end:
                tokens.append(special_tokens["OBJ_END"])
        tokens.append(SEP)

        # Truncate if necessary.
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids += [0] * padding_length
        input_mask += [0] * padding_length
        segment_ids = [0] * max_seq_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids))
    return features

def predict(model, device, eval_dataloader):
    model.eval()
    preds = None  # Initialize as None to avoid TypeError

    with torch.inference_mode(): 
        for input_ids, input_mask, segment_ids in eval_dataloader:
            input_ids, input_mask, segment_ids = (
                input_ids.to(device),
                input_mask.to(device),
                segment_ids.to(device)
            )

            logits = model(input_ids, attention_mask=input_mask)
            logits = logits[0] if isinstance(logits, tuple) else logits
            logits_np = logits.cpu().numpy() 

            if preds is None:
                preds = logits_np
            else:
                preds = np.append(preds, logits_np, axis=0)

    return np.argmax(preds, axis=1), np.max(softmax(preds, axis=1), axis=1)

def get_safe_batch_size(default=8, min_size=2):
    """Dynamically adjust batch size based on available memory"""
    available_mem = psutil.virtual_memory().available / (1024 ** 3)  # Convert to GB
    if available_mem < 2:  # If RAM < 2GB, reduce batch size
        return min_size
    elif available_mem < 4:
        return max(min_size, default // 2)
    return default

class SpanBERT:
    def __init__(self, pretrained_dir, model="spanbert-base-cased", max_seq_length=64, batch_size=8):
        assert os.path.exists(pretrained_dir), "Pre-trained model folder does not exist: {}".format(pretrained_dir)
        self.seed = 42
        self.max_seq_length = max_seq_length
        self.batch_size = get_safe_batch_size(batch_size)
        self.device = torch.device("cpu")
        self.n_gpu = torch.cuda.device_count()
        self._set_seed()
        self.id2label = {i: label for i, label in enumerate(label_list)}
        self.num_labels = len(label_list)
        self.tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=False)

        print("Loading pre-trained SpanBERT from {}".format(pretrained_dir))
        self.classifier = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=self.num_labels)
        self.classifier.to(self.device)

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

    def predict(self, examples):
        features = convert_examples_to_features(examples, self.max_seq_length, self.tokenizer, special_tokens)
        all_input_ids = np.array([f.input_ids for f in features], dtype=np.int32)
        all_input_mask = np.array([f.input_mask for f in features], dtype=np.int32)
        all_segment_ids = np.array([f.segment_ids for f in features], dtype=np.int32)
        data = TensorDataset(
                torch.from_numpy(all_input_ids),
                torch.from_numpy(all_input_mask),
                torch.from_numpy(all_segment_ids)
            )
        dataloader = DataLoader(data, 
            batch_size=self.batch_size, 
            num_workers=0, 
            pin_memory=False,
            persistent_workers=False  # Avoid keeping workers alive unnecessarily
            )
        pred_ids, proba = predict(self.classifier, self.device, dataloader)
        if pred_ids is None:
            return None
        preds = [self.id2label[pred] for pred in pred_ids]
        return list(zip(preds, proba))

if __name__ == "__main__":
    pretrained_dir = os.path.abspath("./pretrained_spanbert")
    bert = SpanBERT(pretrained_dir=pretrained_dir)
    examples = [
        {
            "tokens": "Bill Gates is the founder of Microsoft".split(),
            "subj": ('Bill Gates', "PERSON", (0, 1)),
            "obj": ('Microsoft', "ORGANIZATION", (6, 6))
        },
        {
            "tokens": "Bill Gates is the founder of Microsoft".split(),
            "obj": ('Bill Gates', "PERSON", (0, 1)),
            "subj": ('Microsoft', "ORGANIZATION", (6, 6))
        }
    ]
    preds = bert.predict(examples)

    # Clean up memory
    import gc
    del input_ids, input_mask, segment_ids
    del all_input_ids, all_input_mask, all_segment_ids
    torch.cuda.empty_cache()  # Even on CPU, helps free PyTorch's internal buffers
    gc.collect()  # Force Python garbage collection

    for example, pred in list(zip(examples, preds)):
        example["relation"] = pred
        print(example)
