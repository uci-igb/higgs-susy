#########################################
# Script for training NN on HIGGS dataset.
# Peter Sadowski
# January 10 2014
#
# USAGE:
# Simply specify the parameters in the 'main' function. Then run in python.
# This script will generate two files with a common basename generated from the parameters.
# 1) A .pkl file containing the trained pylearn2 model object.
# 2) A .log file containing information about training.
#
#########################################
import sys
import os
import theano
import pylearn2
import pylearn2.models.mlp as mlp
import pylearn2.training_algorithms
import pylearn2.training_algorithms.sgd
import pylearn2.costs
import pylearn2.train
import pylearn2.termination_criteria
import pickle as pkl
import numpy as np
#from hyperopt import STATUS_OK, STATUS_FAIL

import pylearn2.datasets.physics

def init_train(args):
    # Interpret arguments.
    derived_feat, seed, nhid, width, lrinit, lrdecay, momentum_init, momentum_saturate, wdecay, dropout_include, = args
    derived_feat = derived_feat # string
    seed = int(seed)
    nhid = int(nhid)
    width = int(width)
    lrinit = float(lrinit)
    lrdecay = float(lrdecay)
    momentum_init = float(momentum_init)
    momentum_saturate = int(momentum_saturate)
    wdecay = float(wdecay)
    dropout_include = float(dropout_include) # dropout include probability in top layer.
    
    # Specify output files: log and saved model pkl.
    #idpath = os.path.splitext(os.path.abspath(__file__))[0] # ID for output files.
    idpath = ''
    idpath = idpath + '%s_%d_%d_%d_%0.5f_%0.9f_%0.2f_%d_%f_%0.1f' %(derived_feat, seed, nhid, width, lrinit, lrdecay, momentum_init, momentum_saturate, wdecay, dropout_include) 
    save_path = idpath + '.pkl'
    logfile = idpath + '.log'
    print 'Using=%s' % theano.config.device # Can use gpus. 
    print 'Writing to %s' % logfile
    print 'Writing to %s' % save_path
    sys.stdout = open(logfile, 'w')
    
    # Dataset
    benchmark = 1
    dataset_train = pylearn2.datasets.physics.PHYSICS(which_set='train', benchmark=benchmark, derived_feat=derived_feat) # Smaller set for choosing hyperparameters.
    #dataset_train = pylearn2.datasets.physics.PHYSICS(which_set='train', benchmark=benchmark, derived_feat=derived_feat, start=0, stop=2600000) # Smaller set for choosing hyperparameters.
    dataset_train_monitor = pylearn2.datasets.physics.PHYSICS(which_set='train', benchmark=benchmark, derived_feat=derived_feat, start=0,stop=100000)
    dataset_valid = pylearn2.datasets.physics.PHYSICS(which_set='valid', benchmark=benchmark, derived_feat=derived_feat)
    dataset_test = pylearn2.datasets.physics.PHYSICS(which_set='test', benchmark=benchmark, derived_feat=derived_feat)

    # Model
    nvis = dataset_train.X.shape[1]
    istdev = 1.0/np.sqrt(width)
    layers = []
    for i in range(nhid):
        # Hidden layer i
        layer = mlp.Tanh(layer_name = 'h%d' % i, dim=width,
                        istdev = (istdev if i>0 else 0.1), # First layer should have higher stdev.
                        )
        layers.append(layer)
    #layers.append(mlp.Sigmoid(layer_name='y', dim=1, istdev=istdev/100.0))
    layers.append(mlp.Sigmoid(layer_name='y', dim=1, istdev=0.001))
    model = pylearn2.models.mlp.MLP(layers, nvis=nvis, seed=seed)

    # Cost
    cost = pylearn2.costs.mlp.Default() # Default cost.
    if dropout_include != 1.0:
        # Use dropout cost if specified.
        cost = pylearn2.costs.mlp.dropout.Dropout(
                        input_include_probs={'y': dropout_include},
                        input_scales={'y': 1.0/dropout_include},
                        default_input_include_prob = 1.0,
                        default_input_scale = 1.0)
    if wdecay != 0.0:
        # Add weight decay term if specified.
        cost2 = pylearn2.costs.mlp.WeightDecay(coeffs = [wdecay]*(nhid+1)) # wdecay if specified.
        cost = pylearn2.costs.cost.SumOfCosts(costs=[cost, cost2])
    
    
    # Algorithm
    algorithm = pylearn2.training_algorithms.sgd.SGD(
                    batch_size=100,   # If changed, change learning rate!
                    learning_rate = lrinit, #.05, # In dropout paper=10 for gradient averaged over batch. Depends on batchsize.
                    learning_rule = pylearn2.training_algorithms.learning_rule.Momentum(
                                        init_momentum = momentum_init,
                                    ),
                    monitoring_dataset = {'train':dataset_train_monitor,
                                          'valid':dataset_valid,
                                          'test':dataset_test
                                          },
                    termination_criterion=pylearn2.termination_criteria.Or(criteria=[
                                            pylearn2.termination_criteria.MonitorBased(
                                                channel_name="valid_y_kl",
                                                prop_decrease=0.00001,
                                                N=10),
                                            pylearn2.termination_criteria.EpochCounter(
                                                max_epochs=momentum_saturate)
                                            ]),
                    cost=cost,
                    update_callbacks=pylearn2.training_algorithms.sgd.ExponentialDecay(
                                        decay_factor = lrdecay, #1.0000002 # Decreases by this factor every batch. (1/(1.000001^8000)^100 
                                        min_lr=.000001
                                        )
                )
    # Extensions 
    extensions=[ 
        #pylearn2.train_extensions.best_params.MonitorBasedSaveBest(channel_name='train_y_misclass',save_path=save_path)
        pylearn2.training_algorithms.learning_rule.MomentumAdjustor(
            start=0,
            saturate = momentum_saturate, # 200,500
            final_momentum = 0.99,  # Dropout=.5->.99 over 500 epochs.
            )
        ]
    # Train
    train = pylearn2.train.Train(dataset=dataset_train,
                                 model=model,
                                 algorithm=algorithm,
                                 extensions=extensions,
                                 save_path=save_path,
                                 save_freq=100)

    return train

 
def compute_objective(args):
    '''Initializes neural network with specified arguments and trains.'''    
    # Train network.
    train = init_train(args)
    train.main_loop()
    model = train.model
     
    # Return objective to hyperopt.
    loss = train.model.monitor.channels['valid_y_kl'].val_record[-1]
    return loss
    #return {"loss":loss,"status":STATUS_OK}

if __name__=='__main__':
    # Initialize and train.
    #args = (derived_feat, seed, nhid, width, lrinit, lrdecay, momentum_init, momentum_saturate, wdecay, dropout_include)
    args = ('True', 42, 1, 1000, 0.0005, 1.0000005, 0.9, 200, 0.00000, 1.0)
    compute_objective(args)


