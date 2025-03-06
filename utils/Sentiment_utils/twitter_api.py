import os
from io import StringIO

import pandas as pd
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

def load_tweets_from_csv(uploaded_file):
    if uploaded_file is not None:
        data = StringIO(uploaded_file.getvalue().decode("utf-8"))
        tweet_df = pd.read_csv(data)
        return process_tweets(tweet_df)
    return None

def load_tweets_from_query(query):
    if query:
        tweet_data = fetch_tweets(query)
        if tweet_data:
            tweet_df = pd.DataFrame(tweet_data)
            return process_tweets(tweet_df)
    return None

def load_tweets(option, uploaded_file=None, query=None):
    if option == "Upload CSV":
        return load_tweets_from_csv(uploaded_file)
    elif option == "Fetch Tweets":
        return load_tweets_from_query(query)
    return None