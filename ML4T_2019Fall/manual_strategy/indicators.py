from util import get_data
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


def get_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1),
                       syms=['GOOG', 'AAPL', 'GLD', 'XOM']):
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols

    return prices
