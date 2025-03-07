from typing import List
import pandas as pd
import re

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


def process_tweets(df: pd.DataFrame) -> pd.DataFrame:

    df["processed_text"] = df["text"].apply(nlp_preprocess)
    df["cleaned_text"] = df["text"].apply(basic_cleaning)
    df["bert_orgs"] = df["cleaned_text"].apply(dslim_bert_ner_get_ent)
    return df

def predict_tweet_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df["polarity_predictions"] = df["cleaned_text"].apply(lambda x: finbert(x)[0]['label'])
    return df

def dslim_bert_ner_get_ent(text: str) -> List[str]:
    results = ner_pipeline(text)
    companies = [entity['word'] for entity in results if entity['entity_group'] == "ORG"]
    companies_cleaned = remove_subwords(companies)

    return companies_cleaned