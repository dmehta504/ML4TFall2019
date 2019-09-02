"""Assess a betting strategy.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def author():
    return 'dmehta32'  # replace tb34 with your Georgia Tech username.


def gtid():
    return 902831571  # replace with your GT ID number


def get_spin_result(win_prob):
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def test_code():
    # American Wheel has 18 Reds, 18 Blacks and 2 Greens (0)
    # Probablity of winining on a red/black bet is 18/38 = 0.47368

    # set appropriately to the probability of a win
    win_prob = 18/38.
    np.random.seed(gtid())  # do this only once
    experiment1(win_prob)
    # print(get_spin_result(win_prob))  # test the roulette spin


# add your code here to implement the experiments

def simulationRun(winProb, maxSpin = 1000, targetWinnings = 80):
    i = episode_winnings = 0
    winTracker = np.zeros(maxSpin + 1, dtype=int)

    # Implementing the pseudocode as mentioned in Assignment instructions
    # http://quantsoftware.gatech.edu/Fall_2019_Project_1:_Martingale
    while episode_winnings < targetWinnings:
        won = False
        bet_amount = 1
        while not won:
            i += 1;
            won = get_spin_result(winProb)

            if won:
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                bet_amount *= 2

            winTracker[i] = episode_winnings

            if i == maxSpin:
                return winTracker

    # If episode_winnings reaches 80, forward fill the array with goal
    if i != maxSpin:
        winTracker[i+1:] = targetWinnings
        return winTracker

def experiment1(winProb):
    figure_1 = np.zeros((10, 1001), dtype=int)
    for i in xrange(10):
        figure_1[i, :] = simulationRun(winProb, 10)
    fig, axis = plt.subplot(figsize=(10, 5))
    pd.DataFrame(figure_1.T).plot(title = "Winnings from 10 Simulation Runs", ax=axis)

    axis.set_xlim(0,300)
    axis.set_ylim(-256, 100)
    axis.set_xlabel("Spin")
    axis.set_ylabel("Winnings")

if __name__ == "__main__":
    test_code()
