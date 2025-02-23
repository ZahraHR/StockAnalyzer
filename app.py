import streamlit as st
import pandas as pd
from io import StringIO

# Importer les modules utils
from utils.preprocessing import preprocess, basic_cleaning, dslim_bert_ner_get_ent
from utils.twitter_api import fetch_tweets
from utils.visualization import generate_wordcloud, plot_top_orgs, plot_pie_chart, plot_top5_sentiment_multibar
from transformers import pipeline


finbert = pipeline('sentiment-analysis', model='ProsusAI/finbert')


st.sidebar.title("Stock Analysis")
option = st.sidebar.radio("What do you want to analyze", ("Twitter Sentiment Analysis", "Other"))


def load_tweets():
    tweet_df = None
    if st.sidebar.radio("Choose an option", ("Upload CSV", "Fetch Tweets")) == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            data = StringIO(uploaded_file.getvalue().decode("utf-8"))
            tweet_df = pd.read_csv(data)
    elif st.sidebar.radio("Choose an option", ("Upload CSV", "Fetch Tweets")) == "Fetch Tweets":
        query = st.text_input("Enter your query (e.g., 'Tech Stocks -is:retweet lang:en')", "Tech Stocks -is:retweet lang:en")
        if query:
            tweet_data = fetch_tweets(query)
            if tweet_data:
                tweet_df = pd.DataFrame(tweet_data)
    return tweet_df

if option == "Twitter Sentiment Analysis":
    tweet_df = load_tweets()

    if tweet_df is not None:
        st.session_state.df = tweet_df
        num_rows = st.slider("How many rows to display?", min_value=1, max_value=len(tweet_df), value=5)
        st.write(f"Displaying {num_rows} rows of data:")
        st.write(tweet_df.head(num_rows))

        analysis_option = st.sidebar.radio("Choose an option", ("Visualization", "Predictions"))

        if analysis_option == "Visualization":
            visualize_option = st.sidebar.radio("Choose an option", ("Word Cloud", "Bar Plot"))
            if visualize_option == "Word Cloud":
                st.title("Word Cloud")
                tweet_df["processed_text"] = tweet_df["text"].apply(preprocess)
                generate_wordcloud(tweet_df["processed_text"])
            elif visualize_option == "Bar Plot":
                tweet_df["cleaned_text"] = tweet_df["text"].apply(lambda x: basic_cleaning(x))
                tweet_df["bert_orgs"] = tweet_df.cleaned_text.apply(lambda x: dslim_bert_ner_get_ent(x))
                df_top, fig = plot_top_orgs(tweet_df["bert_orgs"], top_n=5)
                st.write(df_top)
                st.plotly_chart(fig)

        elif analysis_option == "Predictions":
            tweet_df["cleaned_text"] = tweet_df["text"].apply(lambda x: basic_cleaning(x))
            tweet_df["polarity_predictions"] = tweet_df["cleaned_text"].apply(lambda x: finbert(x)[0]['label'])

            fig = plot_pie_chart(tweet_df["polarity_predictions"])
            st.plotly_chart(fig)

            tweet_df["bert_orgs"] = tweet_df.cleaned_text.apply(lambda x: dslim_bert_ner_get_ent(x))
            fig = plot_top5_sentiment_multibar(tweet_df)
            st.plotly_chart(fig)

# streamlit run app.py
