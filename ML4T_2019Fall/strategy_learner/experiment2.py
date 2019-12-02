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

    # Strategy Learner (Impact = 0.1)
    strat_learner = sl.StrategyLearner(verbose=False, impact=0.1)
    strat_learner.addEvidence(symbol, sd, ed, sv=100000)
    df_trades = strat_learner.testPolicy(symbol, sd, ed, sv=100000)
    strat_portvals_imp2 = ms.compute_portvals(df_trades, start_val=100000, commission=0, impact=0.1)


if __name__ == "__main__":
    test_code()