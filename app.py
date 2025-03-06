import pandas as pd
import streamlit as st

from utils.Sentiment_utils.twitter_api import load_tweets
from utils.Sentiment_utils.visualization import get_visualization, display_predictions
from utils.Stock_utils.indicator_utils import get_stock_data, calculate_indicators, plot_stock_indicators

from utils.Stock_utils.data_utils import load_and_scale_data, split_data, create_sequence
from utils.Stock_utils.model_utils import build_lstm_model, train_lstm_model
from utils.Stock_utils.plot_utils import plot_loss, plot_stock_data
from sklearn.metrics import r2_score

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

st.sidebar.title("Stock Analysis")
option = st.sidebar.radio("What do you want to analyze", ("Twitter Sentiment Analysis", "Stock Market Analysis"))

if option == "Twitter Sentiment Analysis":
    tweet_option = st.sidebar.radio("Choose an option", ("Upload CSV", "Fetch Tweets"))
    uploaded_file = None
    query = None
    if tweet_option == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    elif tweet_option == "Fetch Tweets":
        query = st.text_input("Enter your query", "Tech Stocks -is:retweet lang:en")

    tweet_df = load_tweets(tweet_option, uploaded_file, query)

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
    stock_option = st.sidebar.radio("Choose an option:", ["Stock Market Analysis", "Stock Price Prediction"])

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

        if stock_option == "Stock Market Analysis":

            fig = plot_stock_indicators(stock_data, selected_company)

            st.write(f"### Showing data for {selected_company} ({stock_symbol})")
            st.write(stock_data.head())

            st.plotly_chart(fig, use_container_width=True)
        elif stock_option == "Stock Price Prediction":
            st.write(f"### Predicting stock prices for {selected_company} ({stock_symbol})")

            feature_transform, scaler = load_and_scale_data(stock_data)
            X_train, X_val, X_test = split_data(feature_transform)

            train_seq, train_label = create_sequence(X_train)
            val_seq, val_label = create_sequence(X_val)
            test_seq, test_label = create_sequence(X_test)

            model = build_lstm_model((30, 1))
            model, history = train_lstm_model(model, train_seq, train_label, val_seq, val_label)

            fig = plot_loss(history)
            st.plotly_chart(fig, use_container_width=True)

            test_predicted = model.predict(test_seq)
            inv_pred = scaler.inverse_transform(test_predicted)
            inv_actual = scaler.inverse_transform(test_label)

            r2 = r2_score(inv_actual, inv_pred)
            st.write(f"**R² Score:** {r2:.4f}")

            fig = plot_stock_data(stock_data, inv_pred, stock_data.shape[0] - inv_pred.shape[0])
            st.plotly_chart(fig, use_container_width=True)