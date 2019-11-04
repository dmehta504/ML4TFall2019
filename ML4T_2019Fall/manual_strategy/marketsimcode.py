""" Fall 2019 - Project 6 : Manual Strategy
Student Name: Dhruv Mehta (replace with your name)
GT User ID: dmehta32 (replace with your User ID)
GT ID: 902831571 (replace with your GT ID)
"""

import util
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


def get_stock_prices(symbols, dates):
    df_prices = pd.DataFrame(index=dates, columns=symbols)
    for symbol in symbols:
        df_prices[symbol] = util.get_data([symbol], pd.date_range(dates[0], dates[-1]), colname='Adj Close')[symbol]
    return df_prices


def execute_trades(df_trades, df_prices, holdings, start_val, commission, impact):
    # Create a portfolio to track total portfolio value and number of shares of each stock
    portvals = pd.DataFrame(0, index=df_trades.index, columns=["total"])
    total_portfolio_val = start_val

    # Execute the orders and simulate the portfolio
    for date in df_trades.index:
        # If orders are made on that particular date, calculate new values for stocks & portfolio
        if date in df_trades.index:
            orders_made = df_trades.loc[[date]]
            for stock in orders_made.columns:
                number_of_shares = abs(orders_made.iloc[0][stock])
                price_of_stock = df_prices.loc[date, stock]

                if orders_made.iloc[0][stock] > 0:
                    price_of_stock = (1 + impact) * price_of_stock
                    # If shares are bought, subtract cash amount and commission
                    total_portfolio_val = total_portfolio_val - (price_of_stock * number_of_shares) - commission
                elif orders_made.iloc[0][stock] < 0:
                    price_of_stock = (1 - impact) * price_of_stock
                    # If shares are sold, add cash amount and subtract commission
                    total_portfolio_val = total_portfolio_val + (price_of_stock * number_of_shares) - commission

        # Update portfolio value for current date
        for symbol in df_prices.columns:
            # Multiply the number of shares * price of stock for that trading day and add it to the portfolio value
            portvals.loc[date, "total"] += (holdings.loc[date, symbol] * df_prices.loc[date, symbol])

        portvals.loc[date, "total"] += total_portfolio_val

    return portvals


def compute_portvals(df_trades, start_val=1000000, commission=9.95, impact=0.005):
    # Get the historical prices for each symbol in df_trades
    symbols = df_trades.columns.unique().tolist()
    df_prices = get_stock_prices(symbols, df_trades.index)

    # Create dataframe to calculate number of shares of each stock
    holdings = pd.DataFrame(0, index=df_trades.index, columns=df_trades.columns)
    holdings += df_trades
    holdings = holdings.cumsum()

    portvals = execute_trades(df_trades, df_prices, holdings, start_val, commission, impact)
    return portvals


def author():
    return 'dmehta32'


def test_code():
    sd = dt.datetime(2009, 12, 29)
    ed = dt.datetime(2009, 12, 31)
    symbol = ['JPM']
    df_trades = pd.DataFrame([1000, 0, -1000], columns=["JPM"], index=pd.date_range(sd, ed))
    portvals = compute_portvals(df_trades)
    print(portvals)


if __name__ == "__main__":
    test_code()