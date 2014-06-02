Scripts for running Neural Networks on the HIGGS and SUSY datasets.
Author: Peter Sadowski
Date: May 2014

These scripts depend on the Pylearn2 and Theano libraries, and data located at
the UCI Machine Learning Repository.

* Usage *
1) Install Theano and Pylearn2.
http://deeplearning.net/software/pylearn2/

2) Download data from:
http://archive.ics.uci.edu/ml/datasets/HIGGS
http://archive.ics.uci.edu/ml/datasets/SUSY

3) Point PYLEARN2_DATA_PATH towards the location of the data:
export PYLEARN2_DATA_PATH=/home/myhome/higgs-susy/data

4) Copy physics.py to the pylearn2 dataset directory: 
cp physics.py /home/myhome/pylearn2/pylearn2/datasets/

5) A neural network is trained by running a single python script. Results can be observed by examining the log files, or by examining the model that is saved in a pkl file.

6) The AUC (Area Under the Receiver Operator Characteristic Curve) can be computed with the python script computeauc.py

