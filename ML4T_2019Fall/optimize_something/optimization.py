"""MC1-P2: Optimize a portfolio.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Dhruv Mehta (replace with your name)
GT User ID: dmehta32 (replace with your User ID)
GT ID: 902831571 (replace with your GT ID)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1),
                       syms=['GOOG', 'AAPL', 'GLD', 'XOM'], gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    # add code here to find the allocations
    allocs, sr = optimize_sr(syms, prices)
    # allocs = np.round(allocs, 5)

    # Get daily portfolio value
    port_val = compute_daily_portfolio_value(allocs, prices)  # add code here to compute daily portfolio values

    # add code here to compute stats
    sr = sharpeRatio(allocs, prices)
    daily_returns = compute_daily_returns(dailyPortfolioVal=port_val)
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    cr = (daily_returns.ix[-1, :] / daily_returns.ix[0, :]) - 1

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        pass

    return allocs, cr, adr, sddr, sr


def sharpeRatio(allocs, prices):
    k = np.sqrt(252)
    dailyPortfolioVal = compute_daily_portfolio_value(allocs, prices)
    daily_returns = compute_daily_returns(dailyPortfolioVal)
    avg_daily_returns = daily_returns.mean()
    std_daily_returns = daily_returns.std()
    sr = k * (avg_daily_returns / std_daily_returns)

    return sr


def compute_daily_returns(dailyPortfolioVal):
    df_temp = dailyPortfolioVal.copy()
    df_temp[1:] = (dailyPortfolioVal[1:] / dailyPortfolioVal[:-1].values) - 1
    df_temp = df_temp[1:]
    return df_temp


def compute_daily_portfolio_value(allocs, prices):
    # From lecture video 01-07.2
    normed = prices / prices.ix[0, :]
    alloced = normed * allocs
    # position_vals = 1000000 * alloced
    return alloced.sum(axis=1)


def optimize_sr(symbols, prices):
    allocs_guess = np.asarray([1./len(symbols)] * len(symbols))
    bounds = tuple((0.0, 1.0) for _ in range(len(symbols)))
    constraints = ({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)})
    solution = spo.minimize(minimize_sharpe_ratio, allocs_guess, prices, constraints=constraints, bounds=bounds)
    return solution.x, solution.fun


def minimize_sharpe_ratio(allocs, prices):
    return -1 * sharpeRatio(allocs, prices)


def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2009, 1, 1)
    end_date = dt.datetime(2010, 1, 1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date,
                                                        syms=symbols,
                                                        gen_plot=False)

    # Print statistics
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
