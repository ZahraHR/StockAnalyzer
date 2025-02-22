import streamlit as st
import pandas as pd
from io import StringIO

# Importer les modules utils
from utils.preprocessing import preprocess, basic_cleaning, dslim_bert_ner_get_ent
from utils.twitter_api import fetch_tweets
from utils.visualization import generate_wordcloud, plot_top_orgs

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
            tweet_df = pd.read_csv(data)
            st.write(tweet_df.head())

            st.session_state.df = tweet_df

            num_rows = st.slider("How many rows to display?", min_value=1, max_value=len(tweet_df), value=5)
            st.write(f"Displaying {num_rows} rows of data:")
            st.write(tweet_df.head(num_rows))

    elif upload_option == "Fetch Tweets":
        st.title("Fetch Tweets from Twitter")
        query = st.text_input("Enter your query (e.g., 'Tech Stocks -is:retweet lang:en')", "Tech Stocks -is:retweet lang:en")

        if query:
            tweet_data = fetch_tweets(query)
            if tweet_data:
                tweet_df = pd.DataFrame(tweet_data)
                st.write(tweet_df.head())

                st.session_state.df = tweet_df

                num_rows = st.slider("How many rows to display?", min_value=1, max_value=len(tweet_df), value=5)
                st.write(f"Displaying {num_rows} rows of data:")
                st.write(tweet_df.head(num_rows))

    if 'df' in st.session_state and st.session_state.df is not None:
        st.sidebar.title("Visualize data")
        visualize_option = st.sidebar.radio("Choose an option", ("Word Cloud", "Bar Plot"))

        # Word Cloud Visualization
        if visualize_option == "Word Cloud":
            st.title("Word Cloud")
            tweet_df = st.session_state.df  # Use the tweet data from session state
            tweet_df["processed_text"] = tweet_df["text"].apply(preprocess)

            # Generate and display the Word Cloud
            generate_wordcloud(tweet_df["processed_text"])
        elif visualize_option == "Bar Plot":
            st.title("Top Organizations Bar Plot")

            tweet_df["cleaned_text"] = tweet_df["text"].apply(lambda x: basic_cleaning(x))
            tweet_df["bert_orgs"] = tweet_df.cleaned_text.apply(lambda x: dslim_bert_ner_get_ent(x))

            df_top, fig = plot_top_orgs(tweet_df["bert_orgs"], top_n=5)

            st.write(df_top)
            st.plotly_chart(fig)

# streamlit run app.py
