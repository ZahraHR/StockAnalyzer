# Twitter Stock Sentiment Analysis 📊

## Overview

**Twitter Stock Sentiment Analysis** is a data-driven project designed to analyze financial markets using real-time Twitter sentiment and stock market data. The project consists of three main components:

1. **Tweet Sentiment Analysis**: Extracts and analyzes financial tweets to determine overall sentiment and per-company sentiment.
2. **Stock Market Data Analysis**: Retrieves historical stock data from Yahoo Finance, calculates key technical indicators, and forecasts trends using LSTM.
3. **Portfolio Optimization (Coming Soon)**: Future implementation of portfolio optimization strategies.

---

## Features

### **1. Twitter Sentiment Analysis**
- **Retrieve Tweets**:  
  - Import a pre-collected CSV dataset or fetch real-time tweets using the Twitter API.  
  - Requires a Twitter `Bearer Token`, stored in a `.env` file:
  
    ```bash
    TWITTER_BEARER_TOKEN="your_token"
    ```
- **Sentiment Analysis & Visualization**:
  - **Word Clouds**: Highlight the most frequently mentioned terms.  
  - **Bar Plots**: Show the most discussed companies on Twitter.  
  - **Market Sentiment Prediction**: Leverages **FinBERT** for sentiment classification.  
  - **Pie Charts**: Displays sentiment distribution (positive, neutral, negative).  
  - **MultiBar Charts**: Compares sentiment across multiple companies.  

---

### **2. Stock Market Data Analysis (Yahoo Finance)**
- **Select a Stock**: Fetch historical data for a specific stock.  
- **Compute Technical Indicators**:
  - **SMA (Simple Moving Average)**
  - **EMA (Exponential Moving Average)**
  - **RSI (Relative Strength Index)**
  - **MACD (Moving Average Convergence Divergence)**
  - **Signal Line**
- **LSTM-Based Forecasting**: Predicts future stock trends using **Long Short-Term Memory (LSTM) models**.

---

### **3. Portfolio Optimization (Coming Soon)**
- Future implementation of **risk-return optimization** and **portfolio allocation strategies**.

---

## Installation & Setup

### **1. Prerequisites**
Ensure you have the following installed:
- **Python 3.x**
- **pip**
- **Git** (optional, for cloning the repository)

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **3. Run the Application**
```bash
git clone git@github.com:ZahraHR/StockAnalyzer.git
cd StockAnalyzer
source ./venv/bin/activate  # For Mac/Linux users
streamlit run app.py
```