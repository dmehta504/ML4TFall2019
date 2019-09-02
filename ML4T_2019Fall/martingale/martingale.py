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

    #Figure1
    experiment1_figure1(win_prob)
    #Figure2
    experiment1_figure2(win_prob)
    #Figure3
    experiment1_figure3(win_prob)
    #Figure4 & Figure5
    experiment2_figure4_figure5(win_prob)
    # print(get_spin_result(win_prob))  # test the roulette spin


# add your code here to implement the experiments
def simulationRun(winProb, maxSpin = 1000, targetWinnings = 80):
    i = episode_winnings = 0
    winTracker = np.zeros(maxSpin + 1, dtype=np.int)

    # Implementing the pseudocode as mentioned in Assignment instructions
    # http://quantsoftware.gatech.edu/Fall_2019_Project_1:_Martingale
    while episode_winnings < targetWinnings:
        won = False
        bet_amount = 1
        while not won:
            i += 1
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


def simulationRun_withBankRoll(winProb, maxSpin = 1000, targetWinnings = 80, bankRoll = 256):
    i = episode_winnings = 0
    winTracker = np.zeros(maxSpin + 1, dtype=np.int)

    while episode_winnings < targetWinnings:
        won = False
        bet_amount = 1
        while not won:
            i += 1
            won = get_spin_result(winProb)

            if won:
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                #have to account for the case when Bankroll < bet amount
                if bankRoll:
                    bet_amount = min(bet_amount * 2, bankRoll + episode_winnings)
                else:
                    bet_amount *= 2

            winTracker[i] = episode_winnings

            #have to account for the case when Bankroll reaches 0
            if bankRoll + episode_winnings == 0:
                winTracker[i+1:] = episode_winnings #forward fill with episode winnings (i.e -256)
                return winTracker

            if i == maxSpin:
                return winTracker

    # If episode_winnings reaches 80, forward fill the array with goal
    if i != maxSpin:
        winTracker[i+1:] = targetWinnings
        return winTracker


def experiment1_figure1(winProb):
    maxSpins = 1000
    #Create a figure for 10 simulation runs, 10 rows
    figure_1 = np.zeros((10, maxSpins + 1), dtype=np.int)
    for i in range(10):
        figure_1[i, :] = simulationRun(winProb, maxSpin=maxSpins)

    fig, axis = plt.subplots(figsize=(12, 8))
    pd.DataFrame(figure_1.T).plot(title = "Winnings from 10 Simulation Runs", ax=axis)

    axis.set_xlim(0,300)
    axis.set_ylim(-256, 100)
    axis.set_xlabel("Spin")
    axis.set_ylabel("Winnings")
    plt.legend(["Sim1", "Sim2", "Sim3", "Sim4", "Sim5", "Sim6", "Sim7", "Sim8", "Sim9", "Sim10"])
    plt.tight_layout()
    plt.savefig("exp1-fig1.png")


def experiment1_figure2(winProb):
    #Create a figure for 1000 simulation runs, 1000 rows
    figure_2 = np.zeros((1000, 1001), dtype=np.int)
    for i in range(1000):
        figure_2[i, :] = simulationRun(winProb)

    #Calculate mean & Standard deviation of the 1000 runs
    mean = np.mean(figure_2, axis=0)
    std_dev = np.std(figure_2, axis=0)

    fig, axis = plt.subplots(figsize=(20, 10))
    #Create dataframe from the numpy arrays
    df = pd.DataFrame(np.array([mean, mean + std_dev, mean - std_dev]))
    df.ix[0].plot(title = "Winnings from 1000 Simulation Runs", ax=axis) #Plot the mean
    df.ix[1].plot(ax=axis) #Plot the mean + standard deviation
    df.ix[2].plot(ax=axis) #Plot the mean - standard deviation
    #axis.fill_between(df.columns.values, df.ix[1].values, df.ix[2].values, alpha=0.2)

    axis.set_xlim(0, 300)
    axis.set_ylim(-256, 100)
    axis.set_xlabel("Spin")
    axis.set_ylabel("Winnings")
    plt.legend(["Mean", "Mean + StdDev", "Mean - StdDev"])
    plt.tight_layout()
    plt.savefig("exp1-fig2.png")


def experiment1_figure3(winProb):
    # Create a figure for 1000 simulation runs, 1000 rows
    figure_3 = np.zeros((1000, 1001), dtype=np.int)
    for i in range(1000):
        figure_3[i, :] = simulationRun(winProb)

    # Calculate median & Standard deviation of the 1000 runs
    median = np.median(figure_3, axis=0)
    std_dev = np.std(figure_3, axis=0)

    fig, axis = plt.subplots(figsize=(20, 10))
    # Create dataframe from the numpy arrays
    df = pd.DataFrame(np.array([median, median + std_dev, median - std_dev]))
    df.ix[0].plot(title="Winnings from 1000 Simulation Runs", ax=axis)  # Plot the mean
    df.ix[1].plot(ax=axis)  # Plot the median + standard deviation
    df.ix[2].plot(ax=axis)  # Plot the median - standard deviation
    #axis.fill_between(df.columns.values, df.ix[1].values, df.ix[2].values, alpha=0.2)

    axis.set_xlim(0, 300)
    axis.set_ylim(-256, 100)
    axis.set_xlabel("Spin")
    axis.set_ylabel("Winnings")
    plt.legend(["Median", "Median + StdDev", "Median - StdDev"])
    plt.tight_layout()
    plt.savefig("exp1-fig3.png")

    #Creating extra figure to plot variation of standard deviation over 1000 runs
    fig, axis = plt.subplots(figsize=(10, 5))
    pd.DataFrame(std_dev).plot(title="Standard Deviation over 1000 Simulation Runs", ax=axis)
    axis.set_xlim(0, 300)
    axis.set_xlabel("Spin")
    axis.set_ylabel("Standard Deviation")
    plt.legend(["Standard Deviation"])
    plt.tight_layout()
    plt.savefig("exp1-stddevplot.png")


def experiment2_figure4_figure5(winProb):
    figure_4 = np.zeros((1000, 1001), dtype=np.int)
    for i in range(1000):
        figure_4[i, :] = simulationRun_withBankRoll(winProb)

    #Calculate mean, median & Standard deviation of the 1000 simulations
    mean = np.mean(figure_4, axis=0)
    std_dev = np.std(figure_4, axis=0)
    median = np.median(figure_4, axis=0)

    fig, axis = plt.subplots(figsize=(20, 10))
    # Create dataframe from the numpy arrays
    df = pd.DataFrame(np.array([mean, mean + std_dev, mean - std_dev]))
    df.ix[0].plot(title="Winnings from 1000 Simulation Runs with Bankroll", ax=axis)  # Plot the mean
    df.ix[1].plot(ax=axis)  # Plot the mean + standard deviation
    df.ix[2].plot(ax=axis)  # Plot the mean - standard deviation

    axis.set_xlim(0, 300)
    axis.set_ylim(-256, 100)
    axis.set_xlabel("Spin")
    axis.set_ylabel("Winnings")
    plt.legend(["Mean", "Mean + StdDev", "Mean - StdDev"])
    plt.tight_layout()
    plt.savefig("exp2-fig4.png")

    fig, axis = plt.subplots(figsize=(20, 10))
    # Create dataframe from the numpy arrays
    df = pd.DataFrame(np.array([median, median + std_dev, median - std_dev]))
    df.ix[0].plot(title="Winnings from 1000 Simulation Runs with Bankroll", ax=axis)  # Plot the mean
    df.ix[1].plot(ax=axis)  # Plot the median + standard deviation
    df.ix[2].plot(ax=axis)  # Plot the median - standard deviation

    axis.set_xlim(0, 300)
    axis.set_ylim(-256, 100)
    axis.set_xlabel("Spin")
    axis.set_ylabel("Winnings")
    plt.legend(["Median", "Median + StdDev", "Median - StdDev"])
    plt.tight_layout()
    plt.savefig("exp2-fig5.png")

    # Creating extra figure to plot variation of standard deviation over 1000 runs
    fig, axis = plt.subplots(figsize=(10, 5))
    pd.DataFrame(std_dev).plot(title="Standard Deviation over 1000 Simulation Runs", ax=axis)
    axis.set_xlim(0, 300)
    axis.set_xlabel("Spin")
    axis.set_ylabel("Standard Deviation")
    plt.legend(["Standard Deviation"])
    plt.tight_layout()
    plt.savefig("exp2-stddevplot.png")


if __name__ == "__main__":
    test_code()
