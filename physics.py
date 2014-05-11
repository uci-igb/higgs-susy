# Pylearn2 dataset for physics data.
#__authors__ = "Peter Sadowski"
# May 2014

from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.utils import serial
import os
import numpy as np
import pickle as pkl

class PHYSICS(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, 
                 which_set,
                 benchmark,
                 derived_feat=True,
                 version='',
                 seed=None, # Randomize data order if seed is not None
                 start=0, 
                 stop=np.inf):

        self.args = locals()
        path = os.environ['PYLEARN2_DATA_PATH']

        if derived_feat == 'False':
            derived_feat = False
        elif derived_feat == 'True':
            derived_feat = True

        if benchmark == 1:
            inputfile = '$s/HIGGS.pkl' % path
        elif benchmark == 2:
            inputfile = '$s/SUSY.pkl' % path
        
        X = pkl.load(open(inputfile, 'r'))
        y = X[:,0]
        X = X[:,1:]
        X = np.array(X, dtype='float32')
        y = np.array(y, dtype='float32')
        print 'Data loaded: benchmark%d.' % (benchmark)

        # Select a subset of examples.
        if benchmark == 1:
            # HIGGS
            ntrain = 10000000 
            nvalid = 500000 
            ntest  = 500000
        elif bechmark == 2:
            # SUSY
            ntrain = 4000000 
            nvalid = 500000 
            ntest  = 500000
        if which_set == 'train':
            X = X[0:ntrain, :]
            y = y[0:ntrain, :]
        elif which_set == 'valid':
            X = X[ntrain:ntrain+nvalid, :]
            y = y[ntrain:ntrain+nvalid, :]
        elif which_set == 'test':
            X = X[ntrain+nvalid:ntrain+nvalid+ntest, :]
            y = y[ntrain+nvalid:ntrain+nvalid+ntest, :]

        # Decide which feature set to use.
        if benchmark == 1 and derived_feat == 'only':
            # Only the 7 high level features.
            X = X[:, 21:28]
        elif benchmark == 1 and not derived_feat:
            # Only the 21 raw features.
            X = X[:, 0:21]
        elif benchmark == 1 and derived_feat == 'regress':
            # Predict high level features from low level.
            y = X[:, 21:28]
            X = X[:, 0:21]
        elif benchmark == 2 and derived_feat == 'only':
            # Only the 10 high-level features.
            X = X[:, 8:18]
        elif benchmark == 2 and not derived_feat:
            # Only the 8 low-level features.
            X = X[:, 0:8]
        elif benchmark == 3 and derived_feat == 'only':
            # Only the 15 high-level features.
            X = X[:, 10:25]
        elif benchmark == 3 and not derived_feat:
            # Only the 10 raw features.
            X = X[:, 0:10]

        # Randomize data order.
        if seed:
            rng = np.random.RandomState(42)  # reproducible results with a fixed seed
            indices = np.arange(X.shape[0])
            rng.shuffle(indices)
            X = X[indices, :]
            y = y[indices, :]
   
        # Limit number of samples.
        stop = min(stop, X.shape[0])
        X = X[start:stop, :]
        y = y[start:stop, :]

        # Initialize the superclass. DenseDesignMatrix
        super(PHYSICS,self).__init__(X=X, y=y)
        
    def standardize(self, X):
        """
        Standardize each feature:
        1) If data contains negative values, we assume its either normally or uniformly distributed, center, and standardize.
        2) elseif data has large values, we set mean to 1.
        """
        
        for j in range(X.shape[1]):
            vec = X[:, j]
            if np.min(vec) < 0:
                # Assume data is Gaussian or uniform -- center and standardize.
                vec = vec - np.mean(vec)
                vec = vec / np.std(vec)
            elif np.max(vec) > 1.0:
                # Assume data is exponential -- just set mean to 1.
                vec = vec / np.mean(vec)
            X[:,j] = vec
        return X




