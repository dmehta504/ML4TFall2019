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
        if dataX.shape[0] <= self.leaf_size:
            return np.asarray([["Leaf", np.mean(dataY), np.nan, np.nan]])

        elif np.unique(dataY).shape[0] == 1:
            return np.asarray([["Leaf", dataY[0], np.nan, np.nan]])

        else:
            best_factor, best_index = self.select_splitval(dataX, dataY)
            split_val = np.median(dataX[:, best_index])
            check_split_val = dataX[:, best_index] <= split_val

            if np.array_equal(check_split_val, dataX[:, best_index]):
                return np.asarray([["Leaf", np.mean(dataY), np.nan, np.nan]])

            left_tree = self.build_tree(dataX[check_split_val], dataY[check_split_val])
            right_tree = self.build_tree(dataX[not check_split_val], dataY[not check_split_val])
            root = np.asarray([[best_index, split_val, 1, left_tree.shape[0] + 1]])
            return np.vstack((root, left_tree, right_tree))



    def select_splitval(self, factors, results):
        best_factor = 0
        best_index = 0
        for i in range(factors.shape[1]):
            correlation = np.correlate(factors[:, i], results)
            if correlation > best_factor:
                best_factor = correlation
                best_index = i

        return best_factor, best_index


    def query(self, points):
        results = []
        for i in points:
            results.append(self.get_query(self.model, i))

        return np.asarray(results)

    def get_query(self, model, x_val):
        root = model[0]
        if root == "Leaf":
            return root[1]

        if x_val <= root[1]:
            return self.get_query(model[model[2]:, :], x_val)
        else:
            return self.get_query(model[model[3]:, :], x_val)



if __name__=="__main__":
    print("the secret clue is 'zzyzx'")