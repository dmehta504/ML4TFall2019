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
        df1 = sma.rename(columns={symbol: 'SMA'})
        df2 = bb_ratio.rename(columns={symbol: 'BBR'})
        df3 = momentum.rename(columns={symbol: 'MOM'})

        indicators = pd.concat((df1, df2, df3), axis=1)
        indicators.fillna(0, inplace=True)
        indicators = indicators[:-5]
        trainX = indicators.values

        # Constructing trainY
        trainY = []
        for i in range(prices.shape[0] - 5):
            ratio = (prices.ix[i + 5, symbol] - prices.ix[i, symbol]) / prices.ix[i, symbol]
            if ratio > (0.02 + self.impact):
                trainY.append(1)
            elif ratio < (-0.02 - self.impact):
                trainY.append(-1)
            else:
                trainY.append(0)
        trainY = np.array(trainY)

        # Training
        self.learner.addEvidence(trainX, trainY)

        # example use with new colname  		   	  			  	 		  		  		    	 		 		   		 		  
        # volume_all = ut.get_data(syms, dates, colname="Volume")  # automatically adds SPY
        # volume = volume_all[syms]  # only portfolio symbols
        # volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        # if self.verbose: print(volume)

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
        df1 = sma.rename(columns={symbol: 'SMA'})
        df2 = bb_ratio.rename(columns={symbol: 'BBR'})
        df3 = momentum.rename(columns={symbol: 'MOM'})

        indicators = pd.concat((df1, df2, df3), axis=1)
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
