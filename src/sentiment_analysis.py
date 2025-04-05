import os
import streamlit as st
from utils.Sentiment_utils.twitter_api import TwitterSentimentFetcher
from utils.Sentiment_utils.plots import get_visualization, get_predictions_and_figures

from dotenv import load_dotenv

load_dotenv()
bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

def display_visualizations(df):
    visualize_option = st.sidebar.radio("Choose an option", ("Word Cloud", "Bar Plot"))

    top_n = None
    if visualize_option == "Bar Plot":
        top_n = st.sidebar.slider("Number of top organizations to display", min_value=1, max_value=10, value=5)

    result1, result2 = get_visualization(df, visualize_option, top_n)

    if visualize_option == "Word Cloud":
        st.title("Word Cloud")
        st.pyplot(result2)
    elif visualize_option == "Bar Plot":
        st.title("Top Organizations Bar Plot")
        st.write(result1)
        st.plotly_chart(result2)

def display_predictions(df):
    top_n = st.sidebar.slider("Number of top organizations to display", min_value=1, max_value=10, value=5)
    pie_chart_fig, multibar_fig = get_predictions_and_figures(df, top_n)
    st.plotly_chart(pie_chart_fig)
    st.plotly_chart(multibar_fig)

def twitter_sentiment_analysis():
    tweet_option = st.sidebar.radio("Choose an option", ("Upload CSV", "Fetch Tweets"))
    uploaded_file, query = None, None

    if tweet_option == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    elif tweet_option == "Fetch Tweets":
        query = st.text_input("Enter your query", "Tech Stocks -is:retweet lang:en")

    fetcher = TwitterSentimentFetcher(os.getenv("TWITTER_BEARER_TOKEN"))
    tweet_df = fetcher.load_tweets(tweet_option, uploaded_file, query)

    if tweet_df is not None:
        st.session_state.df = tweet_df
        num_rows = st.slider("How many rows to display?", 1, len(tweet_df), 5)
        st.write(f"Displaying {num_rows} rows of data:")
        st.write(tweet_df.head(num_rows))

        analysis_option = st.sidebar.radio("Choose an option", ("Visualization", "Predictions"))
        if analysis_option == "Visualization":
            display_visualizations(tweet_df)
        elif analysis_option == "Predictions":
            display_predictions(tweet_df)
