import matplotlib.pyplot as plt


def plot_loss(history):
    """
    Plots training and validation loss over epochs.
    Returns:
        fig (matplotlib.figure.Figure): The loss plot figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(history.history['loss'], label='Train Loss', color='blue')
    ax.plot(history.history['val_loss'], label='Validation Loss', color='red')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs Validation Loss')
    ax.legend()

    return fig


def plot_stock_data(stock_data, inv_pred, index_stop):
    """
    Plots stock prices and highlights test predictions.

    Parameters:
    - stock_data: DataFrame with stock data (index should be dates).
    - inv_pred: Array of predicted values (inversed back to original scale).
    - index_stop: Index where test predictions start.
    """
    fig = plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data["Close"], label="Actual Close values", color='blue')

    plt.plot(stock_data.index[index_stop:], inv_pred, label="Predictions", color='red')

    plt.xlabel("Date")
    plt.ylabel("Close Value")
    plt.title("Time Series Forecasting")
    plt.legend()

    # Show the plot
    return fig
