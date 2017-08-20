import numpy as np
import RTLearner as rt

class BagLearner(object):

    def __init__(self, learner=rt.RTLearner, kwargs={"leaf_size":1}, bags = 20, boost = False, verbose = False):
        self.leaf_size = kwargs['leaf_size']
        self.learner = learner(self.leaf_size)
        self.bags = bags
        self.boost = boost

    def author(self):
        return 'lyang338'

    def addEvidence(self, Xtrain, Ytrain):
        """
        @summary: Add training data to learner
        """
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain


    def query(self, Xtest):
        """
        @summary: Estimate a set of test points given the model we built.
        """
        num = self.Xtrain.shape[0]
        predictY = []
        Y_sum = np.zeros(Xtest.shape[0])
        #print "test"

        for i in range (self.bags):  # do RTLearner for 20 times
            # Sample with replacement from the same trainX dataset, each bag should contain N items
            # From piazza: https://piazza.com/class/ixt495px8rj3xq?cid=495
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html
            random_sample_index = np.random.choice(self.Xtrain.shape[0], num, replace=True)
            Xtrain_random_sample = self.Xtrain[random_sample_index]
            #print Xtrain_random_sample.shape
            Ytrain_random_sample = self.Ytrain[random_sample_index]
            #print Ytrain_random_sample.shape
            self.learner.addEvidence(Xtrain_random_sample, Ytrain_random_sample)

            Y = self.learner.query(Xtest)
            Y_sum = Y + Y_sum

        predictY = Y_sum / self.bags
        #print "test"
        return predictY


if __name__=="__main__":
    print "the secret clue is 'zzyzx'"

