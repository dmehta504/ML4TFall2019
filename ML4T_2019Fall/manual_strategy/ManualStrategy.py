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
import indicators as ind
import matplotlib.pyplot as plt


def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    # Get the prices for SPY and JPM
    dates = pd.date_range(sd, ed)
    window = 10
    prices_all = util.get_data([symbol], dates)
    prices_all = prices_all / prices_all.iloc[0]  # Normalize the prices
    prices_SPY = prices_all['SPY']
    prices_JPM = prices_all['JPM']

    # Get the indicators
    sma = ind.calculate_sma(prices_JPM)
    upper_band, lower_band, bb_ratio = ind.calculate_bb(prices_JPM)
    momentum = ind.calculate_momentum(prices_JPM, window=10)

    # Set thresholds for the indicators
    sma_threshold = (0.93, 1.06)  # (Low, High)
    bbratio_threshold = (-1.0, 1.0)
    momentum_threshold = (-0.25, 0.25)

    # Make the dataframe for trades
    df_trades = pd.DataFrame(index=prices_SPY.index, columns=[symbol])
    current_holding = 0
    date_last = None
    for date in prices_SPY.index:
        # For SMA, BB and Momentum, our window is 10 days
        # so we need to make no trades until we have the indicators established
        if date_last is None or date <= sd + dt.timedelta(days=window):
            df_trades.loc[date] = 0
            date_last = date
            continue

        if (sma.loc[date] > sma_threshold[1] and momentum.loc[date] > momentum_threshold[1])\
                or (bb_ratio.loc[date] > bbratio_threshold[1]):
            if current_holding == 0:
                df_trades.loc[date] = -1000
                current_holding -= 1000
            elif current_holding == 1000:
                df_trades.loc[date] = -2000
                current_holding -= 2000
            elif current_holding == -1000:
                df_trades.loc[date] = 0

        elif (sma.loc[date] < sma_threshold[0] and momentum.loc[date] < momentum_threshold[0])\
                or (bb_ratio.loc[date] < bbratio_threshold[0]):
            if current_holding == 0:
                df_trades.loc[date] = 1000
                current_holding += 1000
            elif current_holding == -1000:
                df_trades.loc[date] = 2000
                current_holding += 2000
            elif current_holding == 1000:
                df_trades.loc[date] = 0

        else:
            df_trades.loc[date] = 0

        date_last = date

    return df_trades


def author():
    return 'dmehta32'


def test_code():
    # In Sample - Portfolio
    df_trades = testPolicy()
    portvals = ms.compute_portvals(df_trades, start_val=100000, commission=9.95, impact=0.005)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[
            portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    cumulative_return, avg_daily_ret, std_daily_ret, sharpe_ratio = ms.compute_portfolio_stats(portvals)
    print(cumulative_return, avg_daily_ret, std_daily_ret, sharpe_ratio)

    # Benchmark Portfolio
    df_benchmark = pd.DataFrame(index=df_trades.index, columns=["JPM"])
    df_benchmark.loc[df_trades.index] = 0
    df_benchmark.loc[df_trades.index[0]] = 1000  # Buying 1000 shares of JPM
    portvals_benchmark = ms.compute_portvals(df_benchmark, start_val=100000, commission=9.95, impact=0.005)

    # Normalize Portfolio and Benchmark Portfolio
    portvals_norm = portvals / portvals.iloc[0]
    portvals_benchmark = portvals_benchmark / portvals_benchmark.iloc[0]

    # Generate Plot - In Sample
    figure, axis = plt.subplots()
    portvals_norm.plot(ax=axis, color='r')
    portvals_benchmark.plot(ax=axis, color='g')
    plt.title("Comparison of Manual Strategy Portfolio vs Benchmark")
    plt.legend(["Manual Strategy", "Benchmark"])
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    # plt.savefig("ManualStrategy-InSample.png")
    plt.show()

    # Out Sample - Portfolio
    symbol = "JPM"
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    df_trades_os = testPolicy(symbol, sd, ed)
    portvals_os = ms.compute_portvals(df_trades_os, start_val=100000, commission=9.95, impact=0.005)

    # Benchmark Portfolio - Out Sample
    df_benchmark_os = pd.DataFrame(index=df_trades_os.index, columns=["JPM"])
    df_benchmark_os.loc[df_trades_os.index] = 0
    df_benchmark_os.loc[df_trades_os.index[0]] = 1000  # Buying 1000 shares of JPM
    portvals_benchmark_os = ms.compute_portvals(df_benchmark_os, start_val=100000, commission=9.95, impact=0.005)

    # Normalize Portfolio and Benchmark Portfolio
    portvals_norm_os = portvals_os / portvals_os.iloc[0]
    portvals_benchmark_os = portvals_benchmark_os / portvals_benchmark_os.iloc[0]

    # Generate Plot - Out Sample
    figure, axis = plt.subplots()
    portvals_norm_os.plot(ax=axis, color='r')
    portvals_benchmark_os.plot(ax=axis, color='g')
    plt.title("Comparison of Manual Strategy Portfolio vs Benchmark - Out of Sample")
    plt.legend(["Manual Strategy", "Benchmark"])
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    # plt.savefig("ManualStrategy-OutSample.png")
    plt.show()


if __name__ == "__main__":
    test_code()