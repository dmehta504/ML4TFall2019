""" Fall 2019 - Project 6 : Manual Strategy
Student Name: Dhruv Mehta (replace with your name)
GT User ID: dmehta32 (replace with your User ID)
GT ID: 902831571 (replace with your GT ID)
"""
import util
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


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


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input

    # Get the orders from file
    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    start_date = orders.index.min()
    end_date = orders.index.max()

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    portvals_SPY = get_data(['SPY'], pd.date_range(start_date, end_date))
    portvals_SPY = portvals_SPY[['SPY']]  # remove SPY
    dates_index = pd.date_range(start_date, end_date, freq='D')

    # Remove the dates SPY didn't trade on
    for date in dates_index:
        if date not in portvals_SPY.index:
            dates_index = dates_index.drop(date)

    # Get the prices of the stocks that were traded in the orders file, forward fill then back-fill for missing values
    symbols = orders["Symbol"].unique().tolist()
    symbol_dict = {}
    for symbol in symbols:
        symbol_dict[symbol] = get_data([symbol], pd.date_range(start_date, end_date), colname='Adj Close')
        symbol_dict[symbol] = symbol_dict[symbol].resample("D").fillna(method='ffill')
        symbol_dict[symbol] = symbol_dict[symbol].fillna(method='bfill')

    # Create a portfolio to track total portfolio value and number of shares of each stock
    portvals = pd.DataFrame(index=dates_index, columns=["total"] + symbols)

    # Initialize portfolio val at start to starting value, as no money has been made
    total_portfolio_val = start_val
    date_last = None

    # Execute the orders and simulate the portfolio
    for date in dates_index:

        if date_last is None:
            # This is when no trades have been made i.e. first day of trading
            portvals.loc[date, :] = 0
        else:
            # Make a copy of the values of the previous trading day
            portvals.loc[date, :] = portvals.loc[date_last, :]
            portvals.loc[date, "total"] = 0

        # If orders are made on that particular date, calculate new values for stocks & portfolio
        if date in orders.index:
            orders_made = orders.loc[[date]]
            for temp, order in orders_made.iterrows():
                stock = order["Symbol"]
                number_of_shares = order["Shares"]
                buy_or_sell = order["Order"]
                price_of_stock = symbol_dict[stock].loc[date, stock]

                if buy_or_sell == "BUY":
                    price_of_stock = (1 + impact) * price_of_stock
                    # If shares are bought, subtract cash amount and commission
                    total_portfolio_val = total_portfolio_val - (price_of_stock * number_of_shares) - commission
                    portvals.loc[date, stock] += number_of_shares  # Update the number of shares held of that stock
                elif buy_or_sell == "SELL":
                    price_of_stock = (1 - impact) * price_of_stock
                    # If shares are sold, add cash amount and subtract commission
                    total_portfolio_val = total_portfolio_val + (price_of_stock * number_of_shares) - commission
                    portvals.loc[date, stock] -= number_of_shares

        # Update portfolio value for current date
        for symbol in symbols:
            # Multiply the number of shares * price of stock for that trading day and add it to the portfolio value
            portvals.loc[date, "total"] += (portvals.loc[date, symbol] * symbol_dict[symbol].loc[date, symbol])

        portvals.loc[date, "total"] += total_portfolio_val
        date_last = date

    # return the first column containing portfolio values
    portvals = portvals.loc[:, "total"].to_frame()
    return portvals