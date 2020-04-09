import easysurrogate as es
import numpy as np
import h5py
import tkinter as tk
from tkinter import filedialog
from itertools import chain

class Campaign:
    
    def __init__(self, load_data = False, **kwargs):
        
        if load_data:
            if 'file_path' in kwargs:
                file_path = kwargs['file_path']
            else:
                root = tk.Tk()
                root.withdraw()
                file_path = tk.filedialog.askopenfilename(title="Post processing: Open data file", 
                                                          filetypes=(('HDF5 files', '*.hdf5'), 
                                                                    ('All files', '*.*')))
            h5f = h5py.File(file_path, 'r')
    
            h5f = h5py.File(file_path, 'r')
            print('Loaded', h5f.keys())
    
            self.h5f = h5f

    def load_quantized_softmax_network(self):
        #load the neural network
        surrogate = es.methods.ANN(np.random.rand(10), np.random.rand(10))
        surrogate.load_ANN()
        
        #load the auxillary variables (number of time lags, feature list etc)
        print('===============================')
        print('Loading surrogate with parameters:')
        for key in surrogate.aux_vars:
            print(key, ':', surrogate.aux_vars[key])
            vars(self)[key] = surrogate.aux_vars[key]
            
        #check if the reference data is already loaded
        load_data = not hasattr(self, 'h5f')
        
        #make Feature Engineering object
        feat_eng = es.methods.Feature_Engineering(load_data = load_data)
        feat_eng.init_feature_history(self.lags)
        #ugly hack to include info on symmetry in the features
        feat_eng.X_symmetry = self.X_symmetry
        
        if load_data:
            self.h5f = feat_eng.get_hdf5_file()
    
        n_train = surrogate.n_train
        # self.max_lag = np.max(list(chain(*self.lags)))
        y_train = self.h5f[self.target][0:n_train]
    
        #compute target data at next (time) step
        if y_train.ndim == 2:
            y_train = y_train[feat_eng.max_lag:, :]
        elif y_train.ndim == 1:
            y_train = y_train[feat_eng.max_lag:]
        
        print('Binning reference data')
        
        #one-hot encoded training data per B_k
        feat_eng.bin_data(y_train, self.n_bins)
        
        print('done')
        
        #simple sampler to draw random samples from the bins
        sampler = es.methods.SimpleBin(feat_eng)
    
        #features are time lagged, use the data to create initial feature set
        for i in range(feat_eng.max_lag):
            X = [self.h5f[X_i][i].real for X_i in self.feats]
            feat_eng.append_feat(X)

        return surrogate, sampler, feat_eng