import os
from io import StringIO

import pandas as pd
import streamlit as st
import tweepy
from dotenv import load_dotenv

from .predictions import process_tweets

load_dotenv()
bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
client = tweepy.Client(bearer_token=bearer_token)


def fetch_tweets(query: str, max_results=100):
    tweets = client.search_recent_tweets(
        query=query, max_results=max_results, tweet_fields=["text", "created_at"]
    )
    if tweets.data:
        return [{"created_at": tweet.created_at, "text": tweet.text} for tweet in tweets.data]
    return []

def load_tweets():
    tweet_df = None
    option = st.sidebar.radio("Choose an option", ("Upload CSV", "Fetch Tweets"))

    if option == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = StringIO(uploaded_file.getvalue().decode("utf-8"))
            tweet_df = pd.read_csv(data)
            tweet_df = process_tweets(tweet_df)

    elif option == "Fetch Tweets":
        query = st.text_input("Enter your query (e.g., 'Tech Stocks -is:retweet lang:en')", "Tech Stocks -is:retweet lang:en")
        if query:
            tweet_data = fetch_tweets(query)
            if tweet_data:
                tweet_df = pd.DataFrame(tweet_data)
                tweet_df = process_tweets(tweet_df)
    return tweet_df