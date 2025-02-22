import os
import tweepy
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()
bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
client = tweepy.Client(bearer_token=bearer_token)


def fetch_tweets(query: str, max_results=100):
    """Récupérer les tweets selon la requête"""
    tweets = client.search_recent_tweets(
        query=query, max_results=max_results, tweet_fields=["text", "created_at"]
    )
    if tweets.data:
        return [{"created_at": tweet.created_at, "text": tweet.text} for tweet in tweets.data]
    return []
