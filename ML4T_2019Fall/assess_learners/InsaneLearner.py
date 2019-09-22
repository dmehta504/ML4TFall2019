from assess_learners import BagLearner as bg
from assess_learners import LinRegLearner as lrl


class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learner = bg.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, verbose=verbose)

    def author(self):
        return 'dmehta32'

    def addEvidence(self, dataX, dataY):
        self.learner.addEvidence(dataX, dataY)

    def query(self, points):
        return self.learner.query(points)