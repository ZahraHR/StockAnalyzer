import pandas as pd
import streamlit as st

from utils.twitter_api import load_tweets
from utils.visualization import display_visualizations, display_predictions
from utils.stock_utils import get_stock_data, calculate_indicators, plot_stock_indicators



st.sidebar.title("Stock Analysis")
option = st.sidebar.radio("What do you want to analyze", ("Twitter Sentiment Analysis", "Stock Market Analysis"))

if option == "Twitter Sentiment Analysis":
    tweet_df = load_tweets()
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


if option == "Stock Market Analysis":
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)

    company_df = tables[0][["Security", "Symbol"]]
    company_dict = dict(zip(company_df["Security"], company_df["Symbol"]))

    selected_company = st.sidebar.selectbox("Search for a company", options=[""] + list(company_dict.keys()))
    if selected_company:
        stock_symbol = company_dict[selected_company]
        st.sidebar.write(f"**Selected Ticker:** {stock_symbol}")

        stock_data = get_stock_data(stock_symbol)
        stock_data = calculate_indicators(stock_data)

        fig = plot_stock_indicators(stock_data, selected_company)

        # ---- Display Data and Plot ----
        st.write(f"### Showing data for {selected_company} ({stock_symbol})")
        st.write(stock_data.head())

        st.plotly_chart(fig, use_container_width=True)