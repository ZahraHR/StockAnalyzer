import re
from typing import List

from .models import *

def basic_cleaning(text: str) -> str:
    text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9% ]", " ", text)
    text = text.replace("#", "")
    return text

def nlp_preprocess(text: str) -> str:
    text = basic_cleaning(text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text.lower() not in stopwords_set]
    return " ".join(tokens)


def remove_subwords(tokens: List[str]) -> List[str]:
    if tokens:
      cleaned_tokens = [tokens[0]]

      for token in tokens[1:]:
          if token.startswith("##"):
              cleaned_tokens[-1] += token[2:]
          else:
              cleaned_tokens.append(token)
      return cleaned_tokens

    return []

