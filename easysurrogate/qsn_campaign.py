from .campaign import Campaign
import easysurrogate as es
import numpy as np
import h5py
import tkinter as tk
from tkinter import filedialog
from itertools import chain
import pickle

class QSN_Surrogate(Campaign):

    def __init__(**kwargs):
    """
    """        

    def load_quantized_softmax_network(self):
        #load the neural network
        surrogate = es.methods.ANN(np.random.rand(10), np.random.rand(10))
        surrogate.load_ANN()
       
        #make Feature Engineering object
        feat_eng = es.methods.Feature_Engineering()
        feat_eng.init_feature_history(self.lags)
        #ugly hack to include info on symmetry in the features
        feat_eng.X_symmetry = self.X_symmetry

        # #check if the reference data is already loaded
        load_data = not hasattr(self, 'h5f')
        if load_data:
            self.h5f = h5py.File(self.h5f_path, 'r')
            print('Loaded', self.h5f.keys())

        #time-lagged data features are not required when predicting
        #only create time-lagged reference data
        n_train = surrogate.n_train
        y_train = self.h5f[self.target][0:n_train]
    
        #compute target data at next (time) step
        if y_train.ndim == 2:
            y_train = y_train[feat_eng.max_lag:, :]
        elif y_train.ndim == 1:
            y_train = y_train[feat_eng.max_lag:]
        
        print('Binning reference data')
        #one-hot encoded training data per B_k
        feat_eng.bin_data(y_train, self.n_bins)
        
        print('Creating bin sampler')
        #simple sampler to draw random samples from the bins
        sampler = es.methods.SimpleBin(feat_eng)
    
        # print('Initializing features')
        # #features are time lagged, use the data to create initial feature set
        # for i in range(feat_eng.max_lag):
        #     X = [self.h5f[X_i][i].real for X_i in self.feats]
        #     feat_eng.append_feat(X)

        return surrogate, sampler, feat_eng


    def train_quantized_softmax_network(self, feats, target, lags, n_iter, 
                                        n_bins = 10, test_frac = 0.0,
                                        n_layers = 2, n_neurons = 100, 
                                        activation = 'leaky_relu', 
                                        batch_size = 64, lamb = 0.0, save=True,
                                        store_data = False, **kwargs):

        self.feats = feats
        self.target = target
        self.lags = lags
        self.n_bins = n_bins
        
        #Feature engineering object
        feat_eng = es.methods.Feature_Engineering()
        
        n_samples = self.h5f[feats[0]].shape[0]
        n_train = np.int(n_samples*(1.0 - test_frac))
        print('Using first', n_train, 'of', n_samples, 'samples to train QSN')

        #list of features 
        X = [self.h5f[X_i][0:n_train].real for X_i in feats]
        #the data 
        y = self.h5f[target][0:n_train]
        
        #True/False on wether the X features are symmetric arrays or not
        if 'X_symmetry' in kwargs:
            self.X_symmetry = kwargs['X_symmetry']
        else:
            self.X_symmetry = np.zeros(len(X), dtype=bool)
        
        print('Creating time-lagged training data...')
        X_train, y_train = feat_eng.lag_training_data(X, y, lags = lags, 
                                                      X_symmetry=self.X_symmetry)
        print('done')
        
        #one-hot encoded training data per B_k
        feat_eng.bin_data(y_train, n_bins)
        #simple sampler to draw random samples from the bins
        sampler = es.methods.SimpleBin(feat_eng)
        
        #number of softmax layers (one per output)
        n_softmax = y_train.shape[1]
        
        #number of output neurons 
        n_out = n_bins*n_softmax
        
        surrogate = es.methods.ANN(X=X_train, y=feat_eng.y_idx_binned,
                                   n_layers=n_layers, n_neurons=n_neurons, 
                                   n_softmax=n_softmax, n_out=n_out, 
                                   loss='cross_entropy',
                                   activation=activation, batch_size=batch_size,
                                   lamb=lamb, decay_step=10**4, decay_rate=0.9, 
                                   standardize_X=True, standardize_y=False, 
                                   save=False)
        
        print('===============================')
        print('Training Quantized Softmax Network...')
        
        #train network for N_iter mini batches
        surrogate.train(n_iter, store_loss = True)
        
        if save:
            surrogate.save_ANN(store_data = store_data)
            
        self.surrogate = surrogate
            
        return surrogate, sampler

    
    def save_state(self, **kwargs):

        state = {'feats':self.feats, 'target':self.target, 'lags':self.lags,
                 'n_bins':self.n_bins, 'X_symmetry':self.X_symmetry, 
                 'h5f_path':self.h5f_path}
        super().save_state(state, kwargs)
        # if len(file_path) == 0:

        #     root = tk.Tk()
        #     root.withdraw()
            
        #     file = filedialog.asksaveasfile(title="Save campaign state",
        #                                     mode='wb', defaultextension=".pickle")
        # else:
        #     file = open(file_path, 'wb')
        
        # pickle.dump(state, file)
            
        # file.close()