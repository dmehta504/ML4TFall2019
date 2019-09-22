import numpy as np


class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.model = []

    def author(self):
        return 'dmehta32'

    def addEvidence(self, dataX, dataY):
        self.model = self.build_tree(dataX, dataY)

    def build_tree(self, dataX, dataY):
        # Check for termination criteria

        # We have only one sample left (when size = 1), or our sample size is less than the leaf size
        if dataX.shape[0] <= self.leaf_size:
            return np.asarray([["Leaf", np.mean(dataY), np.nan, np.nan]])

        # If all values in dataY are the same, then all values of X will result in the same Y value
        elif np.unique(dataY).shape[0] == 1:
            return np.asarray([["Leaf", dataY[0], np.nan, np.nan]])

        else:
            # Choose a random factor to split the nodes on
            random_factor = np.random.choice(dataX.shape[1])
            split_val = np.median(dataX[:, random_factor])
            check_split_val = dataX[:, random_factor] <= split_val

            # Extra termination check - case when split_val splits all the values to only one side i.e. Left Tree
            if np.array_equal(check_split_val, dataX[:, random_factor]):
                return np.asarray([["Leaf", np.mean(dataY), np.nan, np.nan]])

            # Recursively build the left & right trees
            left_tree = self.build_tree(dataX[check_split_val], dataY[check_split_val])
            right_tree = self.build_tree(dataX[dataX[:, random_factor] > split_val],
                                         dataY[dataX[:, random_factor] > split_val])
            root = np.asarray([[random_factor, split_val, 1, left_tree.shape[0] + 1]], dtype=float)
            return np.vstack((root, left_tree, right_tree))

    def query(self, points):
        results = []
        for i in range(points.shape[0]):
            result = self.get_y_val(points[i, :])
            results.append(float(result))

        return np.asarray(results)

    def get_y_val(self, x_val):
        """
        @summary: Returns the Y value/result based on the decision tree model learned
        @param x_val: The specific query used to find the Y value/result
        """
        i = 0

        # Iterate through the decision tree until we reach the desired leaf
        while self.model[i][0] != "Leaf":
            split_val = self.model[i][1]

            # If value of query is less than the split_val, go in the left tree
            if x_val[int(float(self.model[i][0]))] <= float(split_val):
                # Convert to float then int to avoid runtime errors
                i = i + int(float(self.model[i][2]))
            # Else go in the right tree
            else:
                i = i + int(float(self.model[i][3]))  # This gets the next node we have to search in the right tree

        # If leaf reached, then return the Y-value from our model
        return self.model[i][1]


if __name__=="__main__":
    print("the secret clue is 'zzyzx'")