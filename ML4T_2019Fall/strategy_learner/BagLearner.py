import numpy as np


class BagLearner(object):

    def __init__(self, learner, kwargs, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.list_learners = []
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        for i in range(bags):
            self.list_learners.append(learner(**kwargs))

    def author(self):
        return 'dmehta32'

    def addEvidence(self, dataX, dataY):

        for l in self.list_learners:
            index_array = np.arange(start=0, stop=dataX.shape[0], dtype=int)
            dataX_slice = np.random.choice(index_array, size=dataX.shape[0], replace=True)
            train_x = dataX[dataX_slice]
            train_y = dataY[dataX_slice]
            l.addEvidence(train_x, train_y)

    def query(self, points):
        results = []
        for l in self.list_learners:
            result = l.query(points)
            results.append(result)

        return np.array(results).mean(axis=0)


if __name__=="__main__":
    print("the secret clue is 'zzyzx'")

