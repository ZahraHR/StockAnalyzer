import streamlit as st
from sentiment_analysis import twitter_sentiment_analysis
from stock_analysis import stock_market_analysis


st.sidebar.title("Stock Analysis")
option = st.sidebar.radio("What do you want to analyze", ("Twitter Sentiment Analysis", "Stock Market Analysis"))

if option == "Twitter Sentiment Analysis":
    twitter_sentiment_analysis()
elif option == "Stock Market Analysis":
    stock_market_analysis()
