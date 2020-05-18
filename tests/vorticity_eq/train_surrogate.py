import numpy as np
import easysurrogate as es
import h5py, os
from itertools import chain, product
from scipy.stats import norm, rv_discrete
import matplotlib.pyplot as plt
from matplotlib import animation

####################
# Simulation flags #
####################
train = True             #train the network
make_movie = False       #make a movie (of the training)
predict = False          #predict using the learned SGS term
store = False            #store the prediction 
make_movie_pred = False  #make a movie (of the prediction)

######################
# Network parameters #
######################

#Feature engineering object - loads data file
feat_eng = es.methods.Feature_Engineering(load_data = True)

#get training data
h5f = feat_eng.get_hdf5_file()

#extract features
vort = h5f['w_hat_nm1_LF'][()]
jac = h5f['VgradW_hat_nm1_LF'][()]
sgs = h5f['r_hat_nm1'][()]

# Reshape vectors into vectors
# I = vort.shape[0]; J = vort.shape[1]; K = vort.shape[2]
# vort = vort.reshape([I, J*K])
# jac = jac.reshape([I, J*K])
# sgs = sgs.reshape([I, J*K])

# Reshape features into scalars
vort = vort.flatten()
jac = jac.flatten()
sgs = sgs.flatten()

#Lag features as defined in 'lags'
lags = [[1, 10]]
max_lag = np.max(list(chain(*lags)))

X_train, y_train = feat_eng.lag_training_data([jac], sgs, lags = lags)
# test = np.zeros(3, dtype=y_train.dtype)
# print(test)

#number of bins per B_k
n_bins = 3
#one-hot encoded training data per B_k
feat_eng.bin_data(y_train, n_bins)
#simple sampler to draw random samples from the bins
sampler = es.methods.SimpleBin(feat_eng)

#number of softmax layers (one per output)
n_softmax = 1

#number of output neurons 
n_out = max(np.unique(feat_eng.binnumbers[:, 0]))*n_softmax

#test set fraction
test_frac = 0.5
n_train = np.int(X_train.shape[0]*(1.0 - test_frac))

#train the neural network
if train:
    surrogate = es.methods.ANN(X=X_train[0:n_train], y=feat_eng.y_idx_binned[0:n_train],
                               n_layers=1, n_neurons=20, 
                               n_softmax=n_softmax, n_out=n_out, loss='cross_entropy',
                               activation='leaky_relu', batch_size=512,
                               lamb=0.0, decay_step=10**4, decay_rate=0.9, 
                               standardize_X=True, standardize_y=False, save=True)

    print('=====================================')
    print('Training Quantized Softmax Network...')

    #train network for N_inter mini batches
    N_iter = 10000
    surrogate.train(N_iter, store_loss = True)
    surrogate.compute_misclass_softmax()
#load a neural network from disk
else:
    #first create dummy ANN object    
    surrogate = es.methods.ANN(X=X_train, y=y_train)
    #the load trained network from disk
    surrogate.load_ANN()

X_mean = surrogate.X_mean
X_std = surrogate.X_std
    
#number of features
n_feat = surrogate.n_in
