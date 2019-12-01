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

    # Printing Portfolio statistics
    daily_returns = (ms_port_val / ms_port_val.shift(1)) - 1
    daily_returns = daily_returns[1:]
    cr = (ms_port_val.iloc[-1] / ms_port_val.iloc[0]) - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    a = np.sqrt(252.0)
    sr = (a * (adr)) / sddr

    '''
    print "Manual Strategy Stats"
    print "CR " + str(cr)
    print "Avg of daily returns " + str(adr)
    print "Std deviation of daily returns " + str(sddr)
    print "Sharpe Ratio " + str(sr)
    '''

    # Printing Benchmark statistics
    bench_dr = (bench_port_val / bench_port_val.shift(1)) - 1
    bench_dr = bench_dr[1:]
    cr = (bench_port_val.iloc[-1] / bench_port_val.iloc[0]) - 1
    adr = bench_dr.mean()
    sddr = bench_dr.std()
    a = np.sqrt(252.0)
    sr = (a * (adr)) / sddr

    '''
    print "\nBenchmark Stats"
    print "CR " + str(cr)
    print "Avg of daily returns " + str(adr)
    print "Std deviation of daily returns " + str(sddr)
    print "Sharpe Ratio " + str(sr)
    '''

    # Printing StrategyLearner statistics
    st_dr = (st_port_val / st_port_val.shift(1)) - 1
    st_dr = st_dr[1:]
    cr = (st_port_val.iloc[-1] / st_port_val.iloc[0]) - 1
    adr = st_dr.mean()
    sddr = st_dr.std()
    a = np.sqrt(252.0)
    sr = (a * (adr)) / sddr

    '''
    print "\nStrategy Learner Stats"
    print "CR " + str(cr)
    print "Avg of daily returns " + str(adr)
    print "Std deviation of daily returns " + str(sddr)
    print "Sharpe Ratio " + str(sr)
    '''

    # Plotting charts
    ms_port_val = ms_port_val / ms_port_val[0]
    bench_port_val = bench_port_val / bench_port_val[0]
    st_port_val = st_port_val / st_port_val[0]
    ax = ms_port_val.plot(fontsize=12, color="black", label="Manual Strategy")
    bench_port_val.plot(ax=ax, color="blue", label='Benchmark')
    st_port_val.plot(ax=ax, color="green", label='Strategy Learner')
    plt.title("Experiment 1")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_code()