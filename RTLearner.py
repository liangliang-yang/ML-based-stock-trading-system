import numpy as np
import scipy
import random
import math

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose = False):
        self.leaf_size = leaf_size
        #self.tree = np.empty


    def author(self):
        return 'lyang338'


    def build_tree(self, data):
        dataX = data[:,0:-1]
        dataY = data[:,-1]
        num_samples = data.shape[0]
        #print num_samples
        if num_samples <= self.leaf_size:
            return np.array([["Leaf", np.median(dataY), "NA", "NA"]])
        elif np.all(dataY == dataY[0]):
            return np.array([["Leaf", dataY[0], "NA", "NA"]])
        else:
            #print "here"
            num_features = dataX.shape[1]
            #random_feature_indices = random.sample(xrange(num_features), min(10, num_features))
            split_success = 0
            i = random.randrange(num_features)
            #print "i=", i
            for k in range(10):
                sample1 = random.randrange(num_samples)
                sample2 = random.randrange(num_samples)
                SplitVal = (dataX[sample1, i] + dataX[sample2, i])/2
                left_data = data[data[:,i] <= SplitVal]
                #print "sv is", SplitVal
                #print "ld is", left_data
                right_data = data[data[:,i] > SplitVal]
                if (left_data.shape[0] != 0 and right_data.shape[0] != 0):
                    #print "i is",i
                    #print "sv is", SplitVal
                    #print "ld is", left_data
                    #print "shape[0] is:", left_data.shape[0]
                    split_successt = 1
                    left_tree = self.build_tree(left_data)
                    #print "left is:", left_tree.shape
                    right_tree = self.build_tree(right_data)
                    #print "right is:", right_tree.shape
                    root = np.array([[i, SplitVal, 1, left_tree.shape[0]+1]])
                    #print "root is:", root.shape
                    #node = np.array([left_tree, root, right_tree])
                    node = np.concatenate((root, left_tree, right_tree), axis=0)
                    #print type(node)
                    return node
                #else:
                    #continue
            if (split_success == 0):
                # https://piazza.com/class/ixt495px8rj3xq?cid=528
                # when the iteration time is larger than 10 times feature number, I will assign the median of that y array to that leave and return the leave
                return np.array([["Leaf", np.median(dataY), "NA", "NA"]])

    def addEvidence(self, Xtrain, Ytrain):
        """
        @summary: Add training data to learner
        """
        #return "here3"
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        data = np.column_stack((self.Xtrain,self.Ytrain))
        #print data.shape
        tree = self.build_tree(data)
        self.tree = tree
        #print type(tree)
        #print "here1"
        #return tree

    def query(self, Xtest):
        # get tree from the addEvidence() function
        tree = self.tree

        #print tree

        predictY = []
        for i in range(Xtest.shape[0]):
            #print "i is: ", i
            #for j in range(tree.shape[0]):
            j = 0
            while (j < tree.shape[0]):
                #print "j is: ", j
                if (tree[j][0] != "Leaf"):
                    factorIndex = int(float(tree[j][0]))
                    #print "factor is", int(float(factorIndex))
                    splitVal = float(tree[j][1])
                    #print "split value is", splitVal
                    #print type(splitVal)
                    #print Xtest[i][0]
                    if (Xtest[i][factorIndex] <= splitVal):
                        # in left subtree of node
                        #print tree[j][2]

                        j = j+int(float(tree[j][2]))
                        #print "new 2 j is: ", j
                        #continue
                    else:
                        # in right subtree of node
                        j = j+int(float(tree[j][3]))
                        #print "new 3 j is: ", j
                        #continue
                else:
                    # find leaf
                    #print "label is: ", tree[j][1]
                    predictY.append(float(tree[j][1]))
                    break
                    #continue


        #print predictY
        return predictY





if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
