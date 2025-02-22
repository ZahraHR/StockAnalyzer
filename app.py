import streamlit as st
import pandas as pd
import tweepy
import re
from io import StringIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
nltk.download("punkt")

import os
from dotenv import load_dotenv

load_dotenv()

bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
client = tweepy.Client(bearer_token=bearer_token)

nlp = spacy.load("en_core_web_sm")
stopwords_set = set(stopwords.words("english"))


def basic_cleaning(text: str) -> str:
    text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9% ]", " ", text)
    text = text.replace("#", "")
    return text


def preprocess(text: str) -> str:
    text = basic_cleaning(text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text.lower() not in stopwords_set]
    return " ".join(tokens)


st.sidebar.title("Upload or Fetch Tweets")
upload_option = st.sidebar.radio("Choose an option", ("Upload CSV", "Fetch Tweets"))

if upload_option == "Upload CSV":
    st.title("Upload CSV to Process Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:

        data = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(data)
        st.write(df.head())

        df["processed_text"] = df["text"].apply(preprocess)
        st.write("Processed Text:", df["processed_text"].head())

        wordcloud = WordCloud(max_words=100, width=800, height=400).generate(" ".join(df["processed_text"]))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot()

elif upload_option == "Fetch Tweets":
    st.title("Fetch Tweets from Twitter")
    query = st.text_input("Enter your query (e.g., 'Tech Stocks -is:retweet lang:en')",
                          "Tech Stocks -is:retweet lang:en")

    if query:

        tweets = client.search_recent_tweets(query=query, max_results=100, tweet_fields=["text", "created_at"])

        if tweets.data:
            tweet_data = [{"created_at": tweet.created_at, "text": tweet.text} for tweet in tweets.data]
            tweet_df = pd.DataFrame(tweet_data)
            st.write(tweet_df.head())

            tweet_df["processed_text"] = tweet_df["text"].apply(preprocess)
            st.write("Processed Text:", tweet_df["processed_text"].head())

            wordcloud = WordCloud(max_words=100, width=800, height=400).generate(" ".join(tweet_df["processed_text"]))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot()

# streamlit run app.py
