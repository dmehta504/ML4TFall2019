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
import BagLearner as bl
import RTLearner as rt
import ManualStrategy as mans
import indicators as ind
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

    # Manual Strategy Learner
    ms_trades = mans.testPolicy(symbol, sd, ed, sv=100000)
    ms_portvals = ms.compute_portvals(ms_trades, start_val=100000, commission=9.95, impact=0.005)

    # Benchmark Portfolio
    df_benchmark = pd.DataFrame(index=ms_trades.index, columns=["JPM"])
    df_benchmark.loc[ms_trades.index] = 0
    df_benchmark.loc[ms_trades.index[0]] = 1000  # Buying 1000 shares of JPM
    portvals_benchmark = ms.compute_portvals(df_benchmark, start_val=100000, commission=9.95, impact=0.005)

    # Strategy Learner
    strat_learner = sl.StrategyLearner(verbose=False, impact=0.005)
    strat_learner.addEvidence(symbol, sd, ed, sv=100000)
    df_trades = strat_learner.testPolicy(symbol, sd, ed, sv=100000)
    strat_portvals = ms.compute_portvals(df_trades, start_val=100000, commission=9.95, impact=0.005)

    # Normalize Portfolios
    # ms_portvals = ms_portvals / ms_portvals.iloc[0]
    # strat_portvals = strat_portvals / strat_portvals.iloc[0]
    # portvals_benchmark = portvals_benchmark / portvals_benchmark.iloc[0]

    # In - Sample Stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = ms.compute_portfolio_stats(ms_portvals[ms_portvals.columns[0]])
    cum_ret_bench, avg_daily_ret_bench, std_daily_ret_bench, sharpe_ratio_bench = \
        ms.compute_portfolio_stats(portvals_benchmark[portvals_benchmark.columns[0]])
    cum_ret_strat, avg_daily_ret_strat, std_daily_ret_strat, sharpe_ratio_strat = \
        ms.compute_portfolio_stats(strat_portvals[strat_portvals.columns[0]])

    print(f"Sharpe Ratio of Strategy Learner (In-Sample): {sharpe_ratio_strat}")
    print(f"Sharpe Ratio of Manual Strategy (In-Sample): {sharpe_ratio}")
    print(f"Sharpe Ratio of Benchmark (In-Sample): {sharpe_ratio_bench}")
    print()
    print(f"Cumulative Return of Strategy Learner (In-Sample): {cum_ret_strat}")
    print(f"Cumulative Return of Manual Strategy (In-Sample): {cum_ret}")
    print(f"Cumulative Return of Benchmark (In-Sample): {cum_ret_bench}")
    print()
    print(f"Standard Deviation of Strategy Learner (In-Sample): {std_daily_ret_strat}")
    print(f"Standard Deviation of Manual Strategy (In-Sample): {std_daily_ret}")
    print(f"Standard Deviation of Benchmark (In-Sample): {std_daily_ret_bench}")
    print()
    print(f"Average Daily Return of Strategy Learner (In-Sample): {avg_daily_ret_strat}")
    print(f"Average Daily Return of Manual Strategy (In-Sample): {avg_daily_ret}")
    print(f"Average Daily Return of Benchmark (In-Sample): {avg_daily_ret_bench}")
    print()
    # print(f"Final Portfolio Value of Strategy Learner: {strat_portvals[-1]}")
    # print(f"Final Portfolio Value of Manual Strategy: {ms_portvals[-1]}")
    # print()

    # Plotting charts
    # ms_port_val = ms_port_val / ms_port_val[0]
    # bench_port_val = bench_port_val / bench_port_val[0]
    # st_port_val = st_port_val / st_port_val[0]
    # ax = ms_port_val.plot(fontsize=12, color="black", label="Manual Strategy")
    # bench_port_val.plot(ax=ax, color="blue", label='Benchmark')
    # st_port_val.plot(ax=ax, color="green", label='Strategy Learner')
    # plt.title("Experiment 1")
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Portfolio Value")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    test_code()