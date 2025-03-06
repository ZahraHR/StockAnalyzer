import logging

import streamlit as st
import spacy
from transformers import pipeline

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading SpaCy model: en_core_web_sm...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

stopwords_set = nlp.Defaults.stop_words

ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
logging.getLogger("transformers").setLevel(logging.ERROR)

@st.cache_resource
def load_finbert():
    return pipeline('sentiment-analysis', model='ProsusAI/finbert')

finbert = load_finbert()
