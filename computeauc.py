import os
import theano
import pylearn2
import pyroc # https://github.com/marcelcaraciolo/PyROC
import pickle as pkl
import sys
import pylearn2.datasets.physics as physics

def fprop(model, X, layeridx=-1):
    '''
    Propagate the data through network and return
    activation in layer layeridx. (-1 means last layer)
    '''
    X_theano = model.get_input_space().make_theano_batch()
    Y_theano = X_theano # if layeridx == 0
    if layeridx == -1:
        Y_theano = model.fprop(X_theano)
    else:
        for layer in model.layers[:layeridx]:
            Y_theano = layer.fprop(Y_theano)
    f = theano.function( [X_theano], Y_theano )
    return f(X)

if __name__=='__main__':

    filename = sys.argv[1]
    dataname = 'higgs'

    # Load pylearn2 model object.
    print 'Loading model...'
    model = pkl.load(open(filename,'r'))
    
    # Determine which features were used to train the model from filename.
    if 'all' in filename or 'True' in filename:
        derived_feat = True
    elif 'raw' in filename or 'False' in filename:
        derived_feat = False
    else:
        assert 'only' in filename
        derived_feat = 'only'
    print 'Loading dataset %s...' % dataname
    benchmark = 1 if dataname=='higgs' else 2
    dataset = physics.PHYSICS(benchmark=benchmark, which_set='test', derived_feat=derived_feat)

    # Predict.
    print 'Making predictions...'
    Yhat = fprop(model, dataset.X)
    # Compute area under the ROC curve.
    print 'Computing AUC...'
    auc = pyroc.ROCData(zip(dataset.y, Yhat)).auc()
    error_test = model.monitor.channels['test_y_kl'].val_record[-1]
    print 'AUC=%f, Error=%f, Dataset=%s, Model File=%s' % (auc, error_test, dataname, filename)


