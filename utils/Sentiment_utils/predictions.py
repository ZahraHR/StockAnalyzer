from typing import List
import pandas as pd

from .models import *
from .preprocessing import nlp_preprocess, basic_cleaning, remove_subwords

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