""" Fall 2019 - Project 6 : Manual Strategy
Student Name: Dhruv Mehta (replace with your name)
GT User ID: dmehta32 (replace with your User ID)
GT ID: 902831571 (replace with your GT ID)
"""

import util
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


def get_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), syms=['JPM']):
    # Code from optimization.py
    # Project 2 - Optimize Something
    dates = pd.date_range(sd, ed)
    prices_all = util.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols

    return prices


def calculate_sma(prices):
    sma = prices.rolling(window=10).mean()
    return sma


def calculate_bb(prices):
    std = prices.rolling(window=10).std()
    sma = calculate_sma(prices)
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    bb_ratio = (prices - sma) / (2 * std)
    return upper_band, lower_band, bb_ratio


def calculate_momentum(prices, window):
    momentum = prices.diff(window)/prices.shift(window)
    return momentum


def test_code():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    symbol = ['JPM']
    prices_JPM = get_portfolio(sd, ed, symbol)
    prices = prices_JPM / prices_JPM.iloc[0, :]  # Normalize the prices

    # Calculate the indicators
    sma = calculate_sma(prices)
    upper_band, lower_band, bb_ratio = calculate_bb(prices)
    momentum = calculate_momentum(prices, window=10)

    # Create Dataframe to help plot the various values
    df = pd.concat([prices, sma, upper_band, lower_band, bb_ratio, momentum], axis=1)
    df.columns = ["JPM", "SMA", "Upper Band", "Lower Band", "Bollinger Ratio", "Momentum"]

    # Create the various plots
    # Plot 1 - SMA
    df[["JPM", "SMA"]].plot(figsize=(10, 8))
    plt.savefig("price_sma.png")
    # plt.show()

    # Plot 2 - Bollinger Bands
    df[["JPM", "SMA", "Upper Band", "Lower Band"]].plot(figsize=(15, 20))
    plt.savefig("price_bb.png")
    # plt.show()

    # Plot 3 - Bollinger Band Ratio
    df[["Bollinger Ratio"]].plot(figsize=(10, 8))
    plt.axhline(y=1.0, color='r')
    plt.axhline(y=-1.0, color='r')
    plt.savefig("price_bb_ratio.png")
    # plt.show()

    # Plot 4 - Momentum
    df[["JPM", "Momentum"]].plot(figsize=(10, 8))
    plt.savefig("price_mom.png")
    # plt.show()


def author():
    return 'dmehta32'


if __name__ == "__main__":
    test_code()





