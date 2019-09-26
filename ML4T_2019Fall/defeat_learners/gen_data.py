"""  		   	  			  	 		  		  		    	 		 		   		 		  
template for generating data to fool learners (c) 2016 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
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
import math


# this function should return a dataset (X and Y) that will work  		   	  			  	 		  		  		    	 		 		   		 		  
# better for linear regression than decision trees  		   	  			  	 		  		  		    	 		 		   		 		  
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    num_rows = np.random.randint(10, 1000, dtype=int)  # choose number of rows between 10-1000
    num_cols = np.random.randint(2, 10, dtype=int)   # choose number of columns between 2-10
    X = np.random.random((num_rows, num_cols))  # Fill the X values with random numbers
    Y = np.zeros(num_rows)

    for i in range(0, num_rows):
        # Fill the Y values such that they are correlated to only one column - this ensures a good lin reg fit.
        Y[i] = X[i, :].sum()

    return X, Y


def best4DT(seed=1489683273):
    np.random.seed(seed)
    num_rows = np.random.randint(10, 1000, dtype=int)  # choose number of rows between 10-1000
    num_cols = np.random.randint(2, 10, dtype=int)  # choose number of columns between 2-10
    X = np.random.random((num_rows, num_cols))
    Y = np.zeros(num_rows)

    for i in range(0, num_rows - 1):
        # This fills all
        if i % 3 == 0:
            Y = Y + np.sin(X[i, :].sum()) + np.random.randint(0, 1000)
        else:
            Y = Y - np.tan(X[i, :].sum()) + np.random.randint(0, 1000)

    return X, Y


def author():
    return 'dmehta32'  # Change this to your user ID


if __name__ == "__main__":
    print("they call me Tim.")
