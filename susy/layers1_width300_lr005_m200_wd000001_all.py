#!/auto/igb-libs/linux/centos/6.x/x86_64/pkgs/python/2.7.4/bin/python
#import argparse
import sys
import os
import theano
import pylearn2
import pylearn2.datasets.physics
import pylearn2.training_algorithms.sgd
#import pylearn2.space
import pylearn2.models.mlp as mlp
import pylearn2.train

def init_train():
    # Initialize train object.
    idpath = os.path.splitext(os.path.abspath(__file__))[0] # ID for output files.
    save_path = idpath + '.pkl'

    # Dataset
    #seed = 42
    benchmark = 2
    #derived_feat, nvis = False, 8
    derived_feat, nvis = True, 18
    #derived_feat, nvis = 'only', 10
    dataset_train = pylearn2.datasets.physics.PHYSICS(which_set='train', benchmark=benchmark, derived_feat=derived_feat)
    dataset_train_monitor = pylearn2.datasets.physics.PHYSICS(which_set='train', benchmark=benchmark, derived_feat=derived_feat, start=0,stop=100000)
    dataset_valid = pylearn2.datasets.physics.PHYSICS(which_set='valid', benchmark=benchmark, derived_feat=derived_feat)
    dataset_test = pylearn2.datasets.physics.PHYSICS(which_set='test', benchmark=benchmark, derived_feat=derived_feat)
    
    # Parameters
    momentum_saturate = 200
    
    # Model
    model = pylearn2.models.mlp.MLP(layers=[mlp.Tanh(
                                                layer_name='h0',
                                                dim=300,
                                                istdev=.1),
                                            #istdev=.05 for any intermediates 
                                            mlp.Sigmoid(
                                                layer_name='y',
                                                dim=1,
                                                istdev=.001)
                                           ],
                                    nvis=nvis
                                    )

    # Algorithm
    algorithm = pylearn2.training_algorithms.sgd.SGD(
                    batch_size=100,   # If changed, change learning rate!
                    learning_rate=.05, # In dropout paper=10 for gradient averaged over batch. Depends on batchsize.
                    init_momentum=.9, 
                    monitoring_dataset = {'train':dataset_train_monitor,
                                          'valid':dataset_valid,
                                          'test':dataset_test
                                          },
                    termination_criterion=pylearn2.termination_criteria.Or(criteria=[
                                            pylearn2.termination_criteria.MonitorBased(
                                                channel_name="valid_objective",
                                                prop_decrease=0.00001,
                                                N=10),
                                            pylearn2.termination_criteria.EpochCounter(
                                                max_epochs=momentum_saturate)
                                            ]),
                    cost=pylearn2.costs.cost.SumOfCosts(
                        costs=[pylearn2.costs.mlp.Default(),
                               pylearn2.costs.mlp.WeightDecay(
                                   coeffs=[ .00001, .00001]
                                   )
                               ]
                    ),

                    update_callbacks=pylearn2.training_algorithms.sgd.ExponentialDecay(
                                        decay_factor=1.0000002, # Decreases by this factor every batch. (1/(1.000001^8000)^100 
                                        min_lr=.000001
                                        )
                )
    # Extensions 
    extensions=[ 
        #pylearn2.train_extensions.best_params.MonitorBasedSaveBest(channel_name='train_y_misclass',save_path=save_path)
        pylearn2.training_algorithms.sgd.MomentumAdjustor(
            start=0,
            saturate=momentum_saturate,
            final_momentum=.99  # Dropout=.5->.99 over 500 epochs.
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
    
def train(mytrain):
    # Execute training loop.
    debug = False
    logfile = os.path.splitext(mytrain.save_path)[0] + '.log'
    print 'Using=%s' % theano.config.device # Can use gpus. 
    print 'Writing to %s' % logfile
    print 'Writing to %s' % mytrain.save_path
    sys.stdout = open(logfile, 'w')        
    mytrain.main_loop()


if __name__=='__main__':
    # Initialize and train.
    mytrain = init_train()
    train(mytrain)

