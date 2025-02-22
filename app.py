import streamlit as st
import pandas as pd
from io import StringIO

# Importer les modules utils
from utils.preprocessing import preprocess
from utils.twitter_api import fetch_tweets
from utils.visualization import generate_wordcloud

st.sidebar.title("Stock Analysis")
option = st.sidebar.radio("What do you want to analyze", ("Twitter Sentiment Analysis","Other"))

if option == "Twitter Sentiment Analysis":
    st.sidebar.title("Upload or Fetch Tweets")
    upload_option = st.sidebar.radio("Choose an option", ("Upload CSV", "Fetch Tweets"))

    if upload_option == "Upload CSV":
        st.title("Upload CSV to Process Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            data = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(data)
            st.write(df.head())

            num_rows = st.slider("How many rows to display?", min_value=1, max_value=len(df), value=5)
            st.write(f"Displaying {num_rows} rows of data:")
            st.write(df.head(num_rows))

    elif upload_option == "Fetch Tweets":
        st.title("Fetch Tweets from Twitter")
        query = st.text_input("Enter your query (e.g., 'Tech Stocks -is:retweet lang:en')", "Tech Stocks -is:retweet lang:en")

        if query:
            tweet_data = fetch_tweets(query)
            if tweet_data:
                tweet_df = pd.DataFrame(tweet_data)
                st.write(tweet_df.head())

                num_rows = st.slider("How many rows to display?", min_value=1, max_value=len(tweet_df), value=5)
                st.write(f"Displaying {num_rows} rows of data:")
                st.write(tweet_df.head(num_rows))
# streamlit run app.py
