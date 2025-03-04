from .preprocessing import nlp_preprocess, basic_cleaning, dslim_bert_ner_get_ent
from .models import finbert
def process_tweets(df):
    df["processed_text"] = df["text"].apply(nlp_preprocess)
    df["cleaned_text"] = df["text"].apply(basic_cleaning)
    df["bert_orgs"] = df["cleaned_text"].apply(dslim_bert_ner_get_ent)
    return df

def process_tweet_data(tweet_df):
    tweet_df["polarity_predictions"] = tweet_df["cleaned_text"].apply(lambda x: finbert(x)[0]['label'])
    return tweet_df