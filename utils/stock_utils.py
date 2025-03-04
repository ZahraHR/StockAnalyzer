import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_stock_data(ticker, period="5y"):
    """Télécharge les données boursières d'un ticker spécifique."""
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period=period).sort_index()
    return stock_data

def calculate_indicators(stock_data):
    """Ajoute les indicateurs techniques au dataframe stock_data."""
    # Moving Averages
    stock_data['50_SMA'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['200_SMA'] = stock_data['Close'].rolling(window=200).mean()

    # Exponential Moving Averages (EMA)
    stock_data['50_EMA'] = stock_data['Close'].ewm(span=50, adjust=False).mean()
    stock_data['200_EMA'] = stock_data['Close'].ewm(span=200, adjust=False).mean()

    # Bollinger Bands
    stock_data['Middle_Band'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['Rolling_STD'] = stock_data['Close'].rolling(window=20).std()
    stock_data['Upper_Band'] = stock_data['Middle_Band'] + 2 * stock_data['Rolling_STD']
    stock_data['Lower_Band'] = stock_data['Middle_Band'] - 2 * stock_data['Rolling_STD']

    # RSI Calculation
    stock_data['RSI'] = calculate_rsi(stock_data)

    # MACD Calculation
    stock_data['MACD'] = stock_data['Close'].ewm(span=12, adjust=False).mean() - stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

    return stock_data

def calculate_rsi(data, window=14):
    """Calcule l'indicateur RSI."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def plot_stock_indicators(stock_data, selected_company):

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                            subplot_titles=(f"{selected_company} Stock Price & Indicators",
                                            f"{selected_company} RSI",
                                            f"{selected_company} MACD and Signal Line",
                                            f"{selected_company} Trading Volume"))

    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines',
                                 name='Close Price', line=dict(color='#1f77b4')), row=1, col=1)

    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['50_SMA'], mode='lines',
                                 name='50-Day SMA', line=dict(color='#9467bd')), row=1, col=1)

    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['200_SMA'], mode='lines',
                                 name='200-Day SMA', line=dict(color='#2ca02c')), row=1, col=1)

    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['50_EMA'], mode='lines',
                                 name='50-Day EMA', line=dict(color='#ff7f0e', dash='dot')), row=1, col=1)

    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['200_EMA'], mode='lines',
                                 name='200-Day EMA', line=dict(color='#17becf', dash='dot')), row=1, col=1)

        # ---- Plot RSI ----
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines',
                                 name='RSI', line=dict(color='#9467bd')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#d62728", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#2ca02c", row=2, col=1)

        # ---- Plot MACD and Signal Line ----
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines',
                                 name='MACD', line=dict(color='#000000')), row=3, col=1)
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Signal_Line'], mode='lines',
                                 name='Signal Line', line=dict(color='#d62728')), row=3, col=1)

        # ---- Plot Volume ----
    fig.add_trace(go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            name='Volume',
            marker=dict(color='#1f77b4', opacity=0.9)
        ), row=4, col=1)

        # ---- Layout Customization ----
    fig.update_layout(
            height=1500, width=1200, title_text=f"{selected_company} Stock Market Analysis (Interactive)",
            showlegend=True,
            xaxis_rangeslider_visible=False,
            plot_bgcolor='white',
            font=dict(color='black'),
            hovermode="x unified"  # Synchronise l'affichage des valeurs sur tous les graphes
        )
    return fig