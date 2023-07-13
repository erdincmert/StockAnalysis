#EDA: Exploratory Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def Perform_EDA(data):
    # Data cleaning and preparation
    data['Date'] = pd.to_datetime(data['Date'])
    # Summary statistics
    print("Summary Statistics:")
    print(data.describe())
    print()

    # Data types of columns
    print("Data Types:")
    print(data.dtypes)
    print()

    # Visualizations
    plt.figure(figsize=(10, 6))

    # Line plot of closing prices over time
    plt.subplot(2, 2, 1)
    plt.plot(data['Date'], data['Close'])
    plt.title('Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')

    # Distribution of closing prices
    plt.subplot(2, 2, 2)
    sns.histplot(data['Close'], kde=True)
    plt.title('Distribution of Closing Prices')
    plt.xlabel('Closing Price')
    plt.ylabel('Frequency')

    # Candlestick plot of high, low and close prices
    plt.subplot(2, 2, 3)
    plt.plot(data['Date'], data['High'], label='High')
    plt.plot(data['Date'], data['Low'], label='Low')
    plt.plot(data['Date'], data['Close'], label='Close')
    plt.title('Candlestick Plot')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Volume over time
    plt.subplot(2, 2, 4)
    plt.plot(data['Date'], data['Volume'])
    plt.title('Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()


