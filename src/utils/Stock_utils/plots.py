import matplotlib.pyplot as plt


def plot_loss(history):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(history.history['loss'], label='Train Loss', color='blue')
    ax.plot(history.history['val_loss'], label='Validation Loss', color='red')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs Validation Loss')
    ax.legend()

    return fig


def plot_stock_data(stock_data, inv_pred, index_stop):

    fig = plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data["Close"], label="Actual Close values", color='blue')

    plt.plot(stock_data.index[index_stop:], inv_pred, label="Predictions", color='red')

    plt.xlabel("Date")
    plt.ylabel("Close Value")
    plt.title("Time Series Forecasting")
    plt.legend()

    return fig
