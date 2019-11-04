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
import matplotlib.pyplot as plt


def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    # Get the prices for SPY and JPM
    dates = pd.date_range(sd, ed)
    prices_all = util.get_data([symbol], dates)
    prices_all = prices_all / prices_all.iloc[0]  # Normalize the prices
    prices_SPY = prices_all['SPY']
    prices_JPM = prices_all['JPM']

    df_trades = pd.DataFrame(index=prices_SPY.index, columns=[symbol])
    current_holding = 0
    date_last = None
    for date in prices_SPY.index:
        # This is for the first day of trading
        if date_last is None:
            df_trades.loc[date] = 0
            date_last = date
            continue

        present_price = prices_JPM.loc[date_last]
        future_price = prices_JPM.loc[date]

        # Sell the stock i.e. SHORT
        if future_price < present_price:
            if current_holding == 0:
                df_trades.loc[date_last] = -1000
                current_holding -= 1000
            elif current_holding == 1000:
                df_trades.loc[date_last] = -2000
                current_holding -= 2000
            elif current_holding == -1000:
                df_trades.loc[date_last] = 0

        # Buy the stock i.e. LONG
        elif future_price > present_price:
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

    # For the last day of trading, we have no more future prices to look at
    # Hence, we don't place any orders and assign the value 0
    df_trades.loc[prices_JPM.index[-1]] = 0
    return df_trades


def author():
    return 'dmehta32'


def test_code():
    df_trades = testPolicy()
    portvals = ms.compute_portvals(df_trades, start_val=100000, commission=0, impact=0)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[
            portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"
    # Calculate stats of Portfolio
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = ms.compute_portfolio_stats(portvals)

    # Benchmark Portfolio
    df_benchmark = pd.DataFrame(index=df_trades.index, columns=["JPM"])
    df_benchmark.loc[df_trades.index] = 0
    df_benchmark.loc[df_trades.index[0]] = 1000  # Buying 1000 shares of JPM
    portvals_benchmark = ms.compute_portvals(df_benchmark, start_val=100000, commission=0, impact=0)
    # Calculate stats of Benchmark Portfolio
    cum_ret_bench, avg_daily_ret_bench, std_daily_ret_bench, sharpe_ratio_bench = \
        ms.compute_portfolio_stats(portvals_benchmark[portvals_benchmark.columns[0]])

    # Normalize Portfolio and Benchmark Portfolio
    portvals_norm = portvals / portvals.iloc[0]
    portvals_benchmark = portvals_benchmark / portvals_benchmark.iloc[0]

    # Generate Plot
    figure, axis = plt.subplots()
    portvals_norm.plot(ax=axis, color='r')
    portvals_benchmark.plot(ax=axis, color='g')
    plt.title("Comparison of Theoretically Optimal Portfolio vs Benchmark")
    plt.legend(["Theoretically Optimal Strategy", "Benchmark"])
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.savefig("OptimalStrategy.png")
    # plt.show()

    # Display Portfolio Stats
    start_date = portvals.index[0]
    end_date = portvals.index[-1]

    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of Benchmark: {sharpe_ratio_bench}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of Benchmark: {cum_ret_bench}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of Benchmark: {std_daily_ret_bench}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of Benchmark: {avg_daily_ret_bench}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
