""" Fall 2019 - Project 8 : Strategy Learner
Student Name: Dhruv Mehta (replace with your name)
GT User ID: dmehta32 (replace with your User ID)
GT ID: 902831571 (replace with your GT ID)
"""

import numpy as np
import datetime as dt
import pandas as pd
import util as ut
import StrategyLearner as sl
import marketsimcode as ms
import matplotlib.pyplot as plt


def author():
    return 'dmehta32'


def test_code():
    # Set Seed to have consistency across runs
    np.random.seed(902831571)

    # In - Sample Dates
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    symbol = 'JPM'
    syms = [symbol]
    dates = pd.date_range(sd, ed)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Strategy Learner (Impact = 0)
    strat_learner = sl.StrategyLearner(verbose=False, impact=0)
    strat_learner.addEvidence(symbol, sd, ed, sv=100000)
    df_trades = strat_learner.testPolicy(symbol, sd, ed, sv=100000)
    strat_portvals_zero = ms.compute_portvals(df_trades, start_val=100000, commission=0, impact=0)

    # Strategy Learner (Impact = 0.005)
    strat_learner = sl.StrategyLearner(verbose=False, impact=0.005)
    strat_learner.addEvidence(symbol, sd, ed, sv=100000)
    df_trades = strat_learner.testPolicy(symbol, sd, ed, sv=100000)
    strat_portvals_imp1 = ms.compute_portvals(df_trades, start_val=100000, commission=0, impact=0.005)

    # Strategy Learner (Impact = 0.01)
    strat_learner = sl.StrategyLearner(verbose=False, impact=0.01)
    strat_learner.addEvidence(symbol, sd, ed, sv=100000)
    df_trades = strat_learner.testPolicy(symbol, sd, ed, sv=100000)
    strat_portvals_imp2 = ms.compute_portvals(df_trades, start_val=100000, commission=0, impact=0.01)

    # Normalize Portfolios
    strat_portvals_zero_norm = strat_portvals_zero / strat_portvals_zero.iloc[0]
    strat_portvals_imp1_norm = strat_portvals_imp1 / strat_portvals_imp1.iloc[0]
    strat_portvals_imp2_norm = strat_portvals_imp2 / strat_portvals_imp2.iloc[0]

    # Generate Plots
    figure, axis = plt.subplots()
    strat_portvals_zero_norm.plot(ax=axis, color='r')
    strat_portvals_imp1_norm.plot(ax=axis, color='g')
    strat_portvals_imp2_norm.plot(ax=axis, color='b')
    plt.title("Portfolio Comparisons of Strategy Learner")
    plt.legend(["Impact = 0", "Impact = 0.05", "Impact = 0.01"])
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.savefig("experiment2.png")
    # plt.show()

    # Statistics
    cum_ret_zero, avg_daily_ret_zero, std_daily_ret_zero, sharpe_ratio_zero = ms.compute_portfolio_stats(
        strat_portvals_zero[strat_portvals_zero.columns[0]])
    cum_ret_imp1, avg_daily_ret_imp1, std_daily_ret_imp1, sharpe_ratio_imp1 = \
        ms.compute_portfolio_stats(strat_portvals_imp1[strat_portvals_imp1.columns[0]])
    cum_ret_imp2, avg_daily_ret_imp2, std_daily_ret_imp2, sharpe_ratio_imp2 = \
        ms.compute_portfolio_stats(strat_portvals_imp2[strat_portvals_imp2.columns[0]])

    print(f"Sharpe Ratio of Strategy Learner (Impact = 0): {sharpe_ratio_zero}")
    print(f"Sharpe Ratio of Manual Strategy (Impact = 0.005): {sharpe_ratio_imp1}")
    print(f"Sharpe Ratio of Benchmark (Impact = 0.01): {sharpe_ratio_imp2}")
    print()
    print(f"Cumulative Return of Strategy Learner (Impact = 0): {cum_ret_zero}")
    print(f"Cumulative Return of Manual Strategy (Impact = 0.005): {cum_ret_imp1}")
    print(f"Cumulative Return of Benchmark (Impact = 0.01): {cum_ret_imp2}")
    print()
    # print(f"Standard Deviation of Strategy Learner (Impact = 0): {std_daily_ret_zero}")
    # print(f"Standard Deviation of Manual Strategy (Impact = 0.005): {std_daily_ret_imp1}")
    # print(f"Standard Deviation of Benchmark (Impact = 0.01): {std_daily_ret_imp2}")
    # print()
    # print(f"Average Daily Return of Strategy Learner (Impact = 0): {avg_daily_ret_zero}")
    # print(f"Average Daily Return of Manual Strategy (Impact = 0.005): {avg_daily_ret_imp1}")
    # print(f"Average Daily Return of Benchmark (Impact = 0.01): {avg_daily_ret_imp2}")
    # print()


if __name__ == "__main__":
    test_code()