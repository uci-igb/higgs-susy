Scripts for running Neural Networks on the HIGGS and SUSY datasets.

These scripts depend on the Pylearn2 and Theano libraries, and data located at
the UCI Machine Learning Repository.

* Usage *
1) Install Theano and Pylearn2.

2) Download data from:
http://archive.ics.uci.edu/ml/datasets/HIGGS
http://archive.ics.uci.edu/ml/datasets/SUSY

3) Put data files in pylearn2 datapath.

4) Copy physics.py to the pylearn2 dataset directory: pylearn2/datasets/

5) Run python scripts to train individual NN models. Results can be observed
in the log files.

6) AUC can be computed with the python script computeauc.py
