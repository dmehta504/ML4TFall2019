"""  		   	  			  	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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
"""  		   	  			  	 		  		  		    	 		 		   		 		  

import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import math
import sys
import pandas as pd
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bg


def gtid():
    return 902831571  # replace with your GT ID number


def experiment1_figure1():
    inf = open("Data/Istanbul.csv")
    data_exp1 = np.genfromtxt(inf, delimiter=",")
    data_exp1 = data_exp1[1:, 1:]

    # compute how much of the data is training and testing
    train_rows_exp1 = int(0.6 * data_exp1.shape[0])
    test_rows_exp1 = data_exp1.shape[0] - train_rows_exp1

    # separate out training and testing data
    trainX_exp1 = data_exp1[:train_rows_exp1, 0:-1]
    trainY_exp1 = data_exp1[:train_rows_exp1, -1]
    testX_exp1 = data_exp1[train_rows_exp1:, 0:-1]
    testY_exp1 = data_exp1[train_rows_exp1:, -1]

    # create list of rmse
    in_sample_rmse = []
    out_sample_rmse = []
    leaf_index = np.arange(1, 51)

    # Iterate through various leaf sizes and record the rmse values
    for leaf_size in leaf_index:
        learner_exp1 = dt.DTLearner(leaf_size=leaf_size)
        learner_exp1.addEvidence(trainX_exp1, trainY_exp1)
        predY_exp1 = learner_exp1.query(trainX_exp1)  # get the predictions
        insample_rmse = math.sqrt(((trainY_exp1 - predY_exp1) ** 2).sum() / trainY_exp1.shape[0])
        in_sample_rmse.append(insample_rmse)
        predY_exp1 = learner_exp1.query(testX_exp1)
        outsample_rmse = math.sqrt(((testY_exp1 - predY_exp1) ** 2).sum() / testY_exp1.shape[0])
        out_sample_rmse.append(outsample_rmse)

    # Generate the plots
    fig, axis = plt.subplots(figsize=(12, 8))
    df = pd.DataFrame({"In-Sample RMSE": in_sample_rmse, "Out-Sample RMSE": out_sample_rmse}, index=leaf_index)
    df.plot(ax=axis, title="RMSE vs Leaf Size for DTLearner")
    axis.set_xlabel("Leaf Size")
    axis.set_ylabel("RMSE")
    plt.xticks(np.arange(0, 51, 5))
    plt.axvline(x=10, color='gray', linestyle='--')
    plt.axvline(x=19, color='gray', linestyle='--')
    plt.legend(["In-Sample RMSE", "Out-Sample RMSE", "Leaf-Size=10", "Leaf-Size=19"], loc='lower right')
    plt.tight_layout()
    plt.savefig("exp1-fig1.png")


def experiment2_figure1():
    inf = open("Data/Istanbul.csv")
    data_exp2 = np.genfromtxt(inf, delimiter=",")
    data_exp2 = data_exp2[1:, 1:]

    # compute how much of the data is training and testing
    train_rows_exp2 = int(0.6 * data_exp2.shape[0])
    test_rows_exp1 = data_exp2.shape[0] - train_rows_exp2

    # separate out training and testing data
    trainX_exp2 = data_exp2[:train_rows_exp2, 0:-1]
    trainY_exp2 = data_exp2[:train_rows_exp2, -1]
    testX_exp2 = data_exp2[train_rows_exp2:, 0:-1]
    testY_exp2 = data_exp2[train_rows_exp2:, -1]

    # create list of rmse
    in_sample_rmse = []
    out_sample_rmse = []
    leaf_index = np.arange(1, 51)

    # Iterate through various leaf sizes and record the rmse values
    for leaf_size in leaf_index:
        learner_exp2 = bg.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":leaf_size}, bags=20)
        learner_exp2.addEvidence(trainX_exp2, trainY_exp2)
        predY_exp2 = learner_exp2.query(trainX_exp2)  # get the predictions
        insample_rmse = math.sqrt(((trainY_exp2 - predY_exp2) ** 2).sum() / trainY_exp2.shape[0])
        in_sample_rmse.append(insample_rmse)
        predY_exp2 = learner_exp2.query(testX_exp2)
        outsample_rmse = math.sqrt(((testY_exp2 - predY_exp2) ** 2).sum() / testY_exp2.shape[0])
        out_sample_rmse.append(outsample_rmse)

    # Generate the plots
    fig, axis = plt.subplots(figsize=(12, 8))
    df = pd.DataFrame({"In-Sample RMSE": in_sample_rmse, "Out-Sample RMSE": out_sample_rmse}, index=leaf_index)
    df.plot(ax=axis, title="RMSE vs Leaf Size for BagLearner with 20 Bags using DTLearner")
    axis.set_xlabel("Leaf Size")
    axis.set_ylabel("RMSE")
    plt.xticks(np.arange(0, 51, 5))
    plt.yticks(np.arange(0.000, 0.009, 0.0005))
    # plt.axvline(x=10, color='gray', linestyle='--')
    # plt.axvline(x=19, color='gray', linestyle='--')
    plt.legend(["In-Sample RMSE", "Out-Sample RMSE"], loc='lower right')
    plt.tight_layout()
    plt.savefig("exp2-fig1.png")


def experiment2_figure2():
    inf = open("Data/Istanbul.csv")
    data_exp2 = np.genfromtxt(inf, delimiter=",")
    data_exp2 = data_exp2[1:, 1:]

    # compute how much of the data is training and testing
    train_rows_exp2 = int(0.6 * data_exp2.shape[0])
    test_rows_exp1 = data_exp2.shape[0] - train_rows_exp2

    # separate out training and testing data
    trainX_exp2 = data_exp2[:train_rows_exp2, 0:-1]
    trainY_exp2 = data_exp2[:train_rows_exp2, -1]
    testX_exp2 = data_exp2[train_rows_exp2:, 0:-1]
    testY_exp2 = data_exp2[train_rows_exp2:, -1]

    # create list of rmse
    in_sample_rmse = []
    out_sample_rmse = []
    bag_index = np.arange(1, 21)

    # Iterate through various bag sizes and record the rmse values
    for bag_size in bag_index:
        learner_exp2 = bg.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":1}, bags=bag_size)
        learner_exp2.addEvidence(trainX_exp2, trainY_exp2)
        predY_exp2 = learner_exp2.query(trainX_exp2)  # get the predictions
        insample_rmse = math.sqrt(((trainY_exp2 - predY_exp2) ** 2).sum() / trainY_exp2.shape[0])
        in_sample_rmse.append(insample_rmse)
        predY_exp2 = learner_exp2.query(testX_exp2)
        outsample_rmse = math.sqrt(((testY_exp2 - predY_exp2) ** 2).sum() / testY_exp2.shape[0])
        out_sample_rmse.append(outsample_rmse)

    # Generate the plots
    fig, axis = plt.subplots(figsize=(12, 8))
    df = pd.DataFrame({"In-Sample RMSE": in_sample_rmse, "Out-Sample RMSE": out_sample_rmse}, index=bag_index)
    df.plot(ax=axis, title="RMSE vs Bag Size at Constant Leaf-Size of 1 for BagLearner using DTLearner")
    axis.set_xlabel("Bag Size")
    axis.set_ylabel("RMSE")
    plt.xticks(np.arange(1, 21, 1))
    plt.yticks(np.arange(0.000, 0.009, 0.0005))
    # plt.axvline(x=10, color='gray', linestyle='--')
    # plt.axvline(x=19, color='gray', linestyle='--')
    plt.legend(["In-Sample RMSE", "Out-Sample RMSE"], loc='upper right')
    plt.tight_layout()
    plt.savefig("exp2-fig2.png")


if __name__ == "__main__":

    np.random.seed(gtid())  # do this only once
    experiment1_figure1()
    experiment2_figure1()
    experiment2_figure2()
    # Credits - Piazza Post # 578
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    if sys.argv[1] == "Data/Istanbul.csv":
        data = np.genfromtxt(inf, delimiter=",")
        data = data[1:, 1:]
    else:
        data = np.array([list(map(float, s.strip().split(','))) for s in inf.readlines()])

    # compute how much of the data is training and testing  		   	  			  	 		  		  		    	 		 		   		 		  
    train_rows = int(0.6* data.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		   	  			  	 		  		  		    	 		 		   		 		  

    # separate out training and testing data  		   	  			  	 		  		  		    	 		 		   		 		  
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    print(f"{testX.shape}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"{testY.shape}")  		   	  			  	 		  		  		    	 		 		   		 		  

    # create a learner and train it  		   	  			  	 		  		  		    	 		 		   		 		  
    # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    # learner = dt.DTLearner(leaf_size=1, verbose=False)
    # learner = rt.RTLearner(leaf_size=1, verbose=False)
    learner = bg.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":1}, bags=1)
    learner.addEvidence(trainX, trainY)  # train it
    print(learner.author())  		   	  			  	 		  		  		    	 		 		   		 		  

    # evaluate in sample  		   	  			  	 		  		  		    	 		 		   		 		  
    predY = learner.query(trainX) # get the predictions  		   	  			  	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print("In sample results")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		   	  			  	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(predY, y=trainY)  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		   	  			  	 		  		  		    	 		 		   		 		  

    # evaluate out of sample  		   	  			  	 		  		  		    	 		 		   		 		  
    predY = learner.query(testX) # get the predictions  		   	  			  	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print("Out of sample results")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		   	  			  	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(predY, y=testY)  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")

