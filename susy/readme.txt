This directory contains scripts to train the NNs in on the SUSY dataset. The filename describes the network to train, including:
1) The number of hidden layers
2) The number of hidden units in each hidden layer
3) Whether weight decay (wd) or dropout (do) was used as a regularization technique.
4) The feature set used ('raw' features, 'only' the high level features, or 'all' features)


To change the random seed that is used to initialize the NN parameters, change
the 'seed' value in the script.


