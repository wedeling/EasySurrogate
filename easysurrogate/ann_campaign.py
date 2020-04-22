import easysurrogate as es
import numpy as np
import h5py
import tkinter as tk
from tkinter import filedialog
from itertools import chain
import pickle

class ANN_Campaign:


    def __init__(self, load_data = False, load_state = False, **kwargs):
        
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
    
            self.h5f_path = file_path
            self.h5f = h5f
            
        if load_state:
            self.load_state()

    def load_artificial_neural_network(self):
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

        # print('Initializing features')
        # #features are time lagged, use the data to create initial feature set
        # for i in range(feat_eng.max_lag):
        #     X = [self.h5f[X_i][i].real for X_i in self.feats]
        #     feat_eng.append_feat(X)

        return surrogate, feat_eng


    def train_artificial_neural_network(self, feats, target, lags, n_iter, 
                                        test_frac = 0.0,
                                        n_layers = 2, n_neurons = 100, 
                                        activation = 'leaky_relu', 
                                        batch_size = 64, lamb = 0.0, save=True,
                                        store_data = False, **kwargs):

        self.feats = feats
        self.target = target
        self.lags = lags
        
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
        
        #number of output neurons 
        n_out = y_train.shape[1]
        
        surrogate = es.methods.ANN(X=X_train, y=y_train,
                                   n_layers=n_layers, n_neurons=n_neurons, 
                                   n_out=n_out, 
                                   loss='squared',
                                   activation=activation, batch_size=batch_size,
                                   lamb=lamb, decay_step=10**4, decay_rate=0.9, 
                                   standardize_X=True, standardize_y=True, 
                                   save=False)
        
        print('===============================')
        print('Training Artificial Neural Network...')
        
        #train network for N_inter mini batches
        surrogate.train(n_iter, store_loss = True)
        
        if save:
            surrogate.save_ANN(store_data = store_data)
            
        self.surrogate = surrogate
            
        return surrogate

    
    def save_state(self, file_path = ""):

        state = {'feats':self.feats, 'target':self.target, 'lags':self.lags,
                 'X_symmetry':self.X_symmetry, 'h5f_path':self.h5f_path}
        
        if len(file_path) == 0:

            root = tk.Tk()
            root.withdraw()
            
            file = filedialog.asksaveasfile(title="Save campaign state",
                                            mode='wb', defaultextension=".pickle")
        else:
            file = open(file_path, 'wb')
        
        pickle.dump(state, file)
            
        file.close()


    def load_state(self, file_path = ""):
      
        #select file via GUI is file_path is not specified
        if len(file_path) == 0:

            root = tk.Tk()
            root.withdraw()
            
            file_path = filedialog.askopenfilename(title="Open campaign state", 
                                           filetypes=(('pickle files', '*.pickle'), 
                                                      ('All files', '*.*')))

        print('Loading state from', file_path)

        file = open(file_path, 'rb')
        state = pickle.load(file)
        file.close()
        
        for key in state:
            print('Loading:', key)
            vars(self)[key] = state[key]