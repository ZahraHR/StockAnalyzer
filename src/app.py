import streamlit as st


st.sidebar.title("Stock Analysis")
option = st.sidebar.radio("What do you want to analyze", ("Twitter Sentiment Analysis", "Stock Market Analysis"))

if option == "Twitter Sentiment Analysis":
    from sentiment_analysis import twitter_sentiment_analysis
    twitter_sentiment_analysis()
elif option == "Stock Market Analysis":
    from stock_analysis import stock_market_analysis
    stock_market_analysis()
