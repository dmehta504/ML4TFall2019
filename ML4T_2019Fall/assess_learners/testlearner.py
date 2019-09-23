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
import time


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


def experiment3_figure1():
    inf = open("Data/Istanbul.csv")
    data_exp3 = np.genfromtxt(inf, delimiter=",")
    data_exp3 = data_exp3[1:, 1:]

    # compute how much of the data is training and testing
    train_rows_exp3 = int(0.6 * data_exp3.shape[0])
    test_rows_exp3 = data_exp3.shape[0] - train_rows_exp3

    # separate out training and testing data
    trainX_exp3 = data_exp3[:train_rows_exp3, 0:-1]
    trainY_exp3 = data_exp3[:train_rows_exp3, -1]
    testX_exp3 = data_exp3[train_rows_exp3:, 0:-1]
    testY_exp3 = data_exp3[train_rows_exp3:, -1]
    trainY_exp3_mean = np.mean(trainY_exp3)
    testY_exp3_mean = np.mean(testY_exp3)

    # create list of r2 scores - 1 - (ss_res/ss_tot)
    # Formula for R2 Score - Credits : https://en.wikipedia.org/wiki/Coefficient_of_determination
    in_sample_r2score_dt = []
    out_sample_r2score_dt = []
    in_sample_r2score_rt = []
    out_sample_r2score_rt = []
    leaf_index = np.arange(1, 21)

    # Iterate through various bag sizes and record the rmse values
    for leaf_size in leaf_index:
        learner_exp3_dt = dt.DTLearner(leaf_size=leaf_size)
        learner_exp3_rt = rt.RTLearner(leaf_size=leaf_size)
        learner_exp3_dt.addEvidence(trainX_exp3, trainY_exp3)
        learner_exp3_rt.addEvidence(trainX_exp3, trainY_exp3)

        # In - Sample Calculations - DTLearner
        predY_exp3_dt = learner_exp3_dt.query(trainX_exp3)  # get the predictions
        ss_res_dt = ((trainY_exp3 - predY_exp3_dt) ** 2).sum()
        ss_tot_dt = ((trainY_exp3 - trainY_exp3_mean) ** 2).sum()
        r2score_dt = 1 - (ss_res_dt/ss_tot_dt)
        in_sample_r2score_dt.append(r2score_dt)

        # In - Sample Calculation - RTLearner
        predY_exp3_rt = learner_exp3_rt.query(trainX_exp3)  # get the predictions
        ss_res_rt = ((trainY_exp3 - predY_exp3_rt) ** 2).sum()
        ss_tot_rt = ((trainY_exp3 - trainY_exp3_mean) ** 2).sum()
        r2score_rt = 1 - (ss_res_rt / ss_tot_rt)
        in_sample_r2score_rt.append(r2score_rt)

        # Out - Sample Calculations - DTLearner
        predY_exp3_dt = learner_exp3_dt.query(testX_exp3)  # get the predictions
        ss_res_dt = ((testY_exp3 - predY_exp3_dt) ** 2).sum()
        ss_tot_dt = ((testY_exp3 - testY_exp3_mean) ** 2).sum()
        r2score_dt = 1 - (ss_res_dt / ss_tot_dt)
        out_sample_r2score_dt.append(r2score_dt)

        # Out - Sample Calculations - RTLearner
        predY_exp3_rt = learner_exp3_rt.query(testX_exp3)  # get the predictions
        ss_res_rt = ((testY_exp3 - predY_exp3_rt) ** 2).sum()
        ss_tot_rt = ((testY_exp3 - testY_exp3_mean) ** 2).sum()
        r2score_rt = 1 - (ss_res_rt / ss_tot_rt)
        out_sample_r2score_rt.append(r2score_rt)

    # Generate the plots
    fig, axis = plt.subplots(figsize=(12, 8))
    df = pd.DataFrame({"In-Sample R2Score - DTLearner": in_sample_r2score_dt,
                       "Out-Sample R2Score - DTLearner": out_sample_r2score_dt,
                       "In-Sample R2Score - RTLearner": in_sample_r2score_rt,
                       "Out-Sample R2Score - RTLearner": out_sample_r2score_rt}, index=leaf_index)

    df.plot(ax=axis, title="R-Square(R2 Score) vs LeafSize Comparison for RTLearner & DTLearner")
    axis.set_xlabel("Leaf Size")
    axis.set_ylabel("R2 Score")
    plt.xticks(np.arange(1, 21, 1))
    plt.yticks(np.arange(0.000, 1.100, 0.1))
    plt.legend(["In-Sample R2Score - DTLearner", "Out-Sample R2Score - DTLearner",
                "In-Sample R2Score - RTLearner", "Out-Sample R2Score - RTLearner"], loc='upper right')
    plt.tight_layout()
    plt.savefig("exp3-fig1.png")


def experiment3_figure2():
    inf = open("Data/Istanbul.csv")
    data_exp3 = np.genfromtxt(inf, delimiter=",")
    data_exp3 = data_exp3[1:, 1:]

    # create list of time values
    time_taken_dt = []
    time_taken_rt = []
    size_index = np.arange(1, data_exp3.shape[0], 20)  # Create an index of how much data to split

    # Iterate through various leaf sizes and record the rmse values
    for size in size_index:
        learner_exp3_dt = dt.DTLearner(leaf_size=1)
        learner_exp3_rt = rt.RTLearner(leaf_size=1)
        trainX_exp3 = data_exp3[:size, 0:-1]
        trainY_exp3 = data_exp3[:size, -1]

        # Calculate time taken to learn - DTLearner
        start_time = time.time()
        learner_exp3_dt.addEvidence(trainX_exp3, trainY_exp3)
        end_time = time.time()
        time_taken = end_time - start_time
        time_taken_dt.append(time_taken)

        # Calculate time taken to learn - RTLearner
        start_time = time.time()
        learner_exp3_rt.addEvidence(trainX_exp3, trainY_exp3)
        end_time = time.time()
        time_taken = end_time - start_time
        time_taken_rt.append(time_taken)


    # Generate the plots
    fig, axis = plt.subplots(figsize=(12, 8))
    df = pd.DataFrame({"Time-Taken DTLearner": time_taken_dt, "Time-Taken RTLearner": time_taken_rt}, index=size_index)
    df.plot(ax=axis, title="Comparison of time taken to train between DTLearner and RTLearner")
    axis.set_xlabel("Size of Training Set")
    axis.set_ylabel("Time-Taken(secs)")
    plt.yticks(np.arange(0.00, 0.40, 0.02))
    plt.xticks(np.arange(0, data_exp3.shape[0], 50))
    plt.legend(["Time-Taken DTLearner", "Time-Taken RTLearner"], loc='upper left')
    plt.tight_layout()
    plt.savefig("exp3-fig2.png")


if __name__ == "__main__":

    np.random.seed(gtid())  # do this only once

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

    # print(f"{testX.shape}")
    # print(f"{testY.shape}")

    # create a learner and train it  		   	  			  	 		  		  		    	 		 		   		 		  
    # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    # learner = dt.DTLearner(leaf_size=1, verbose=False)
    # learner = rt.RTLearner(leaf_size=1, verbose=False)
    learner = bg.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":1}, bags=1)
    learner.addEvidence(trainX, trainY)  # train it
    # print(learner.author())

    # evaluate in sample  		   	  			  	 		  		  		    	 		 		   		 		  
    predY = learner.query(trainX) # get the predictions  		   	  			  	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    # print()
    # print("In sample results")
    # print(f"RMSE: {rmse}")
    c = np.corrcoef(predY, y=trainY)  		   	  			  	 		  		  		    	 		 		   		 		  
    # print(f"corr: {c[0,1]}")

    # evaluate out of sample  		   	  			  	 		  		  		    	 		 		   		 		  
    predY = learner.query(testX) # get the predictions  		   	  			  	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])  		   	  			  	 		  		  		    	 		 		   		 		  
    # print()
    # print("Out of sample results")
    # print(f"RMSE: {rmse}")
    c = np.corrcoef(predY, y=testY)  		   	  			  	 		  		  		    	 		 		   		 		  
    # print(f"corr: {c[0,1]}")

    # Code for generating plots used in report
    experiment1_figure1()
    experiment2_figure1()
    experiment2_figure2()
    experiment3_figure1()
    experiment3_figure2()

