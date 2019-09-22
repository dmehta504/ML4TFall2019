import numpy as np

class DTLearner(object):

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
            # Find the best factor to split the nodes on
            best_factor, best_index = self.select_splitval(dataX, dataY)
            split_val = np.median(dataX[:, best_index])
            check_split_val = dataX[:, best_index] <= split_val

            # Extra termination check - case when split_val splits all the values to only one side i.e. Left Tree
            if np.array_equal(check_split_val, dataX[:, best_index]):
                return np.asarray([["Leaf", np.mean(dataY), np.nan, np.nan]])

            # Recursively build the left & right trees
            left_tree = self.build_tree(dataX[check_split_val], dataY[check_split_val])
            right_tree = self.build_tree(dataX[dataX[:, best_index] > split_val],
                                         dataY[dataX[:, best_index] > split_val])
            root = np.asarray([[best_index, split_val, 1, left_tree.shape[0] + 1]], dtype=float)
            return np.vstack((root, left_tree, right_tree))

    def select_splitval(self, factors, results):
        best_factor = 0
        best_index = 0
        for i in range(factors.shape[1]):
            correlation = np.corrcoef(factors[:, i], results)[0, 1]
            if abs(correlation) > best_factor:
                best_factor = abs(correlation)
                best_index = i

        return best_factor, best_index

    def query(self, points):
        results = []
        for i in range(points.shape[0]):
            result = self.get_y_val(points[i, :])
            results.append(float(result))

        return np.asarray(results)

    def get_y_val(self, x_val):
        i = 0
        while self.model[i][0] != "Leaf":
            split_val = self.model[i][1]
            if x_val[int(float(self.model[i][0]))] <= float(split_val):
                i = i + int(float(self.model[i][2]))
            else:
                i = i + int(float(self.model[i][3]))

        return self.model[i][1]


if __name__=="__main__":
    print("the secret clue is 'zzyzx'")