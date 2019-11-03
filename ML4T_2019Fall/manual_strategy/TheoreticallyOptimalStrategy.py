""" Fall 2019 - Project 6 : Manual Strategy
Student Name: Dhruv Mehta (replace with your name)
GT User ID: dmehta32 (replace with your User ID)
GT ID: 902831571 (replace with your GT ID)
"""

import util
import numpy as np
import pandas as pd
import datetime as dt
import marketsimcode as ms


def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    # Get the prices for SPY and JPM
    dates = pd.date_range(sd, ed)
    prices_all = util.get_data([symbol], dates)
    prices_all = prices_all / prices_all.iloc[0]  # Normalize the prices
    prices_SPY = prices_all['SPY']
    prices_JPM = prices_all['JPM']
    daily_returns = compute_daily_returns(prices_JPM)

    df_trades = pd.DataFrame(index=prices_SPY.index, columns=[symbol])
    current_holding = 0
    date_last = None
    for date in prices_SPY.index:
        # This is for the first day of trading
        if date_last is None:
            df_trades.loc[date] = 0
            date_last = date
            continue

        if daily_returns.loc[date] < 0:
            if current_holding == 0:
                df_trades.loc[date_last] = -1000
                current_holding -= 1000
            elif current_holding == 1000:
                df_trades.loc[date_last] = -2000
                current_holding -= 2000
            elif current_holding == -1000:
                df_trades.loc[date_last] = 0

        elif daily_returns.loc[date] > 0:
            if current_holding == 0:
                df_trades.loc[date_last] = 1000
                current_holding += 1000
            elif current_holding == -1000:
                df_trades.loc[date_last] = 2000
                current_holding += 2000
            elif current_holding == 1000:
                df_trades.loc[date_last] = 0

        else:
            df_trades.loc[date_last] = 0

        date_last = date

    return df_trades


def compute_daily_returns(prices):
    daily_returns = prices / prices.shift(1) - 1
    return daily_returns


def test_code():
    df_trades = testPolicy()
    portvals = ms.compute_portvals(df_trades, start_val=100000, commission=0, impact=0)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[
            portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    cumulative_return, avg_daily_ret, std_daily_ret, sharpe_ratio = ms.compute_portfolio_stats(portvals)
    print(cumulative_return, avg_daily_ret, std_daily_ret, sharpe_ratio)


if __name__ == "__main__":
    test_code()
