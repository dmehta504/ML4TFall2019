""" Fall 2019 - Project 6 : Manual Strategy
Student Name: Dhruv Mehta (replace with your name)
GT User ID: dmehta32 (replace with your User ID)
GT ID: 902831571 (replace with your GT ID)
"""

import util
import numpy as np
import pandas as pd
import datetime as dt


def compute_portfolio_stats(portvals):
    # Calculate cumulative return
    cumulative_return = (portvals / portvals[0]) - 1
    cumulative_return = cumulative_return[-1]

    # Calculate daily returns
    daily_return = portvals.copy()
    daily_return[1:] = (portvals[1:] / portvals[:-1].values) - 1
    avg_daily_ret = daily_return[1:].mean()

    # Calculate standard deviation of daily returns
    std_daily_ret = daily_return[1:].std()

    # Calculate Sharpe Ratio
    k = np.sqrt(252)
    sharpe_ratio = k * (avg_daily_ret/std_daily_ret)

    return cumulative_return, avg_daily_ret, std_daily_ret, sharpe_ratio


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

