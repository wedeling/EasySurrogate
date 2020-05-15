import numpy as np
import easysurrogate as es

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
X_train, y_train = feat_eng.lag_training_data([jac], sgs, lags = lags)

#number of bins per B_k
n_bins = 10
#one-hot encoded training data per B_k
feat_eng.bin_data(y_train, n_bins)
#simple sampler to draw random samples from the bins
sampler = es.methods.SimpleBin(feat_eng)