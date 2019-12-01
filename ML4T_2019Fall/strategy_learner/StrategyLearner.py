"""  		   	  			  	 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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

import numpy as np
import datetime as dt
import pandas as pd
import util as ut
import random
import BagLearner as bl
import RTLearner as rt
import ManualStrategy as mans
import indicators as ind
import marketsimcode as ms


class StrategyLearner(object):

    # constructor  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size":5}, bags=20)

    def addEvidence(self, symbol="IBM", \
                    sd=dt.datetime(2008, 1, 1), \
                    ed=dt.datetime(2009, 1, 1), \
                    sv=10000):

        # add your code to do learning here  		   	  			  	 		  		  		    	 		 		   		 		  
        # example usage of the old backward compatible util function  		   	  			  	 		  		  		    	 		 		   		 		  
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		   	  			  	 		  		  		    	 		 		   		 		  
        prices = prices_all[syms]  # only portfolio symbols  		   	  			  	 		  		  		    	 		 		   		 		  
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print(prices)

        # Get Indicators
        sma = ind.calculate_sma(prices)
        bb_upper, bb_lower, bb_ratio = ind.calculate_bb(prices)
        window = 10
        momentum = ind.calculate_momentum(prices, window)
        indicators = pd.concat((sma, bb_ratio, momentum), axis=1)
        indicators.columns = ['SMA', 'BBR', 'MOM']
        indicators.fillna(0, inplace=True)
        indicators = indicators[:-5]
        trainX = indicators.values

        # Create the Training Set for Y Values based on the indicators
        # We use market variance as 2% and use N = 5 days
        N = 5
        YBUY = 0.02 + self.impact
        YSELL = -0.02 - self.impact
        prices = prices / prices.iloc[0]
        trainY = np.zeros(prices.shape[0] - N)  # Normalize Prices
        for t in range(prices.shape[0] - N):
            # ret = (price[t+N]/price[t]) - 1.0
            ret = (prices.ix[t + N, symbol] / prices.ix[t, symbol]) - 1.0
            if ret > YBUY:
                trainY[t] = 1  # LONG
            elif ret < YSELL:
                trainY[t] = -1  # SHORT
            else:
                trainY[t] = 0  # CASH

        # Feed our bag learner the training data to learn a strategy
        self.learner.addEvidence(trainX, trainY)

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol="IBM", \
                   sd=dt.datetime(2009, 1, 1), \
                   ed=dt.datetime(2010, 1, 1), \
                   sv=10000):

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        # prices_SPY = prices_all['SPY']  # only SPY, for comparison later

        # Get Indicators
        sma = ind.calculate_sma(prices)
        bb_upper, bb_lower, bb_ratio = ind.calculate_bb(prices)
        window = 10
        momentum = ind.calculate_momentum(prices, window)
        indicators = pd.concat((sma, bb_ratio, momentum), axis=1)
        indicators.columns = ['SMA', 'BBR', 'MOM']
        indicators.fillna(0, inplace=True)
        indicators = indicators[:-5]
        testX = indicators.values

        # Querying the learner for testY
        testY = self.learner.query(testX)

        # Constructing trades DataFrame
        trades = prices_all[syms].copy()
        trades.loc[:] = 0
        flag = 0
        for i in range(testY.shape[0]):
            if flag == 0:
                if testY[i] > 0:
                    trades.values[i, :] = 1000
                    flag = 1
                elif testY[i] < 0:
                    trades.values[i, :] = -1000
                    flag = -1

            elif flag == 1:
                if testY[i] < 0:
                    trades.values[i, :] = -2000
                    flag = -1
                elif testY[i] == 0:
                    trades.values[i, :] = -1000
                    flag = 0

            else:
                if testY[i] > 0:
                    trades.values[i, :] = 2000
                    flag = 1
                elif testY[i] == 0:
                    trades.values[i, :] = 1000
                    flag = 0

        if flag == -1:
            trades.values[prices.shape[0] - 1, :] = 1000
        elif flag == 1:
            trades.values[prices.shape[0] - 1, :] = -1000

        if self.verbose: print(
            type(trades))  # it better be a DataFrame!
        if self.verbose: print(trades)
        if self.verbose: print(prices_all)

        return trades

    def author(self):
        return 'dmehta32'


if __name__ == "__main__":
    print("One does not simply think up a strategy")
