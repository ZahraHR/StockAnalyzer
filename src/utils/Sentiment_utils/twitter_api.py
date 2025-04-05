from io import StringIO

import pandas as pd
import tweepy

from .predictions import process_tweets


class TwitterSentimentFetcher:
    def __init__(self, bearer_token: str):
        self.client = tweepy.Client(bearer_token=bearer_token)

    def fetch_tweets(self, query: str, max_results: int = 100) -> pd.DataFrame:
        tweets = self.client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=["text", "created_at"]
        )
        data = [{"created_at": t.created_at, "text": t.text} for t in tweets.data] if tweets.data else []
        return pd.DataFrame(data)

    def load_from_csv(self, uploaded_file) -> pd.DataFrame:
        data = StringIO(uploaded_file.getvalue().decode("utf-8"))
        return pd.read_csv(data)

    def load_tweets(self, option: str, uploaded_file=None, query=None) -> pd.DataFrame:
        df = None
        if option == "Upload CSV":
            if uploaded_file:
                df = self.load_from_csv(uploaded_file)
        elif option == "Fetch Tweets":
            if query:
                df = self.fetch_tweets(query)

        if df is not None:
            return process_tweets(df)
        return None
