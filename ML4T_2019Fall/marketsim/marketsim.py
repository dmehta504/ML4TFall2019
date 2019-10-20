"""MC2-P1: Market simulator.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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
  		   	  			  	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		   	  			  	 		  		  		    	 		 		   		 		  
GT User ID: dmehta32 (replace with your User ID)
GT ID: 902831571 (replace with your GT ID)
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


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


def author():
    return 'dmehta32'


def test_code():
    # this is a helper function you can use to test your code  		   	  			  	 		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		   	  			  	 		  		  		    	 		 		   		 		  
    # Define input parameters  		   	  			  	 		  		  		    	 		 		   		 		  

    of = "./orders/orders-01.csv"
    sv = 1000000

    # Process orders  		   	  			  	 		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[
            portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.  		   	  			  	 		  		  		    	 		 		   		 		  
    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2, 0.01, 0.02, 1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2, 0.01, 0.02, 1.5]

    # Compare portfolio against $SPX  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
