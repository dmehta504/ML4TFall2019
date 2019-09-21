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
            split_val = self.select_splitval(dataX, dataY)
            return None

    def query(self, points):
        pass


if __name__=="__main__":
    print("the secret clue is 'zzyzx'")