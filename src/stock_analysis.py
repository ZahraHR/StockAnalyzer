import streamlit as st
import pandas as pd
from sklearn.metrics import r2_score
from utils.Stock_utils.indicators import get_stock_data, calculate_indicators, plot_stock_indicators
from utils.Stock_utils.data import load_and_scale_data, split_data, create_sequence
from utils.Stock_utils.models import build_lstm_model, train_lstm_model
from utils.Stock_utils.plots import plot_loss, plot_stock_data

def display_stock_market_analysis(stock_data, selected_company, stock_symbol):
    fig = plot_stock_indicators(stock_data, selected_company)
    st.write(f"### Showing data for {selected_company} ({stock_symbol})")
    st.write(stock_data.head())
    st.plotly_chart(fig, use_container_width=True)

def display_stock_price_prediction(stock_data, selected_company, stock_symbol):
    st.write(f"### Predicting stock prices for {selected_company} ({stock_symbol})")

    feature_transform, scaler = load_and_scale_data(stock_data)
    X_train, X_val, X_test = split_data(feature_transform)

    train_seq, train_label = create_sequence(X_train)
    val_seq, val_label = create_sequence(X_val)
    test_seq, test_label = create_sequence(X_test)

    model = build_lstm_model((30, 1))
    model, history = train_lstm_model(model, train_seq, train_label, val_seq, val_label)

    st.plotly_chart(plot_loss(history), use_container_width=True)

    test_predicted = model.predict(test_seq)
    inv_pred = scaler.inverse_transform(test_predicted)
    inv_actual = scaler.inverse_transform(test_label)

    r2 = r2_score(inv_actual, inv_pred)
    st.write(f"**R² Score:** {r2:.4f}")

    st.plotly_chart(plot_stock_data(stock_data, inv_pred, stock_data.shape[0] - inv_pred.shape[0]), use_container_width=True)

def stock_market_analysis():
    stock_option = st.sidebar.radio("Choose an option:", ["Stock Market Analysis", "Stock Price Prediction"])

    # Charger la liste des entreprises du S&P 500
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
            display_stock_market_analysis(stock_data, selected_company, stock_symbol)
        elif stock_option == "Stock Price Prediction":
            display_stock_price_prediction(stock_data, selected_company, stock_symbol)
