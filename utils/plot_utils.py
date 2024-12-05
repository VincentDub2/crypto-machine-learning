import matplotlib.pyplot as plt

def plot_predictions(real_prices, predicted_prices, dates=None, title="Price Predictions"):
    """
    Plot real and predicted prices for comparison.

    Parameters:
    - real_prices (array-like): Array of real prices.
    - predicted_prices (array-like): Array of predicted prices.
    - dates (array-like, optional): Dates corresponding to the prices.
    - title (str): Title of the plot.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    if dates is not None:
        plt.plot(dates, real_prices, label="Real Prices")
        plt.plot(dates, predicted_prices, label="Predicted Prices")
    else:
        plt.plot(real_prices, label="Real Prices")
        plt.plot(predicted_prices, label="Predicted Prices")

    plt.legend()
    plt.title(title)
    plt.xlabel("Time" if dates is None else "Date")
    plt.ylabel("Price")
    plt.grid()
    plt.show()
