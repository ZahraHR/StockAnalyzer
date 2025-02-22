import re
import spacy

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

