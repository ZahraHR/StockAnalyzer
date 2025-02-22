import re
import spacy
from transformers import pipeline
from typing import List
import logging

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading SpaCy model: en_core_web_sm...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Use spaCy's stopwords set
stopwords_set = nlp.Defaults.stop_words

def basic_cleaning(text: str) -> str:
    """Nettoyage de base du texte"""
    text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9% ]", " ", text)
    text = text.replace("#", "")
    return text

def preprocess(text: str) -> str:
    """Appliquer le prétraitement au texte"""
    text = basic_cleaning(text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text.lower() not in stopwords_set]
    return " ".join(tokens)


def remove_subwords(tokens: List[str]) -> List[str]:
    """Remove subword tokens (e.g., ##) and combine them with the previous token."""
    if tokens:
      cleaned_tokens = [tokens[0]]

      for token in tokens[1:]:
          if token.startswith("##"):
              cleaned_tokens[-1] += token[2:]
          else:
              cleaned_tokens.append(token)
      return cleaned_tokens

    return []

ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
logging.getLogger("transformers").setLevel(logging.ERROR)
def dslim_bert_ner_get_ent(text: str):
    """Extraction d'entités avec le pipeline BERT de HuggingFace"""

    results = ner_pipeline(text)

    companies = [entity['word'] for entity in results if entity['entity_group'] == "ORG"]

    # Post-process to combine subword tokens (like "B" and "##YD")
    companies_cleaned = remove_subwords(companies)

    return companies_cleaned

