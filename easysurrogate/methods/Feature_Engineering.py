import numpy as np
from itertools import chain

"""
===============================================================================
CLASS FOR FEATURE ENGINEERING SUBROUTINES
===============================================================================
"""

class Feature_Engineering:
        
    def __init__(self, X, y):
        
        self.X = X
        self. y = y
        
    def standardize_data(self, X_only = False, y_only = False):
        """
        Normalize the training data
        """
        
        X_mean = np.mean(self.X, axis = 0)
        X_std = np.std(self.X, axis = 0)
        
        y_mean = np.mean(self.y, axis = 0)
        y_std = np.std(self.y, axis = 0)
        
        if X_only:
            self.X = (self.X - X_mean)/X_std
        elif y_only:
            self.y = (self.y - y_mean)/y_std
        else:
            self.X = (self.X - X_mean)/X_std
            self.y = (self.y - y_mean)/y_std
       
        return self.X, self.y

    def lag_training_data(self, X, lags):
        """    
        Create time-lagged supervised training data X, y
        
        Parameters:
            X: features. Either an array of dimension (n_samples, n_features)
               or a list of arrays of dimension (n_samples, n_features)
               
            y: training target. Array of dimension (n_samples, n_outputs)
            
            lags: list of lists, containing the integer values of lags
                  Example: if X=[X_1, X_2] and lags = [[1], [1, 2]], the first 
                  feature array X_1 is lagged by 1 (time) step and the second
                  by 1 and 2 (time) steps.
                  
        Returns:
            X_train, y_trains (arrays), of lagged features and target data. Every
            row of X_train is one (time) lagged feature vector. Every row of y_train
            is a target vector at the next (time) step
        """
        
        #compute the max number of lags in lags
        lags_flattened = list(chain(*lags))
        max_lag = np.max(lags_flattened)
        
        #total number of data samples
        n_samples = self.y.shape[0]
        
        #if X is one array, add it to a list anyway
        if type(X) == np.ndarray:
            tmp = []
            tmp.append(X)
            X = tmp
        
        #compute target data at next (time) step
        if self.y.ndim == 2:
            y_train = self.y[max_lag:, :]
        elif self.y.ndim == 1:
            y_train = self.y[max_lag:]
        else:
            print("Error: y must be of dimension (n_samples, ) or (n_samples, n_outputs)")
            return
        
        #a lag list must be specified for every feature in X
        if len(lags) != len(X):
            print('Error: no specified lags for one of the featutes in X')
            return
        
        #compute the lagged features
        C = []
        idx = 0
        for X_i in X:
          
            for lag in lags[idx]:
                begin = max_lag - lag
                end = n_samples - lag
                
                if X_i.ndim == 2:
                    C.append(X_i[begin:end, :])
                elif X_i.ndim == 1:
                    C.append(X_i[begin:end])
                else:
                    print("Error: X must contains features of dimension (n_samples, ) or (n_samples, n_features)")
                    return
            idx += 1
                
        #C is a list of lagged features, turn into a single array X_train
        X_train = C[0]
        
        if X_train.ndim == 1:
            X_train = X_train.reshape([y_train.shape[0], 1])    
        
        for X_i in C[1:]:
            
            if X_i.ndim == 1:
                X_i = X_i.reshape([y_train.shape[0], 1])
            
            X_train = np.append(X_train, X_i, axis=1)
           
        #initialize the storage of features
        self.init_feature_history(lags)
                
        return X_train, y_train
    
    def init_feature_history(self, lags):
        """
        Initialize the feat_history dict. This dict keeps track of the features
        arrays that were used up until 'max_lag' steps ago.
        
        Parameters:             
            
            lags: list of lists, containing the integer values of lags
                  Example: if X=[X_1, X_2] and lags = [[1], [1, 2]], the first 
                  feature array X_1 is lagged by 1 (time) step and the second
                  by 1 and 2 (time) steps.
        """
        self.lags = lags
#        self.max_lag = np.max(list(chain(*lags)))
#        
#        #if there is no lag (feratures at the same time level as the data)
#        #set max_lag = 1 manually in order to use append_feat and get_feat_history
#        #subroutines
#        if self.max_lag == 0:
#            self.max_lag = 1
                
        self.feat_history = {}
        
        #the number of feature arrays that make up the total input feature vector
        self.n_feat_arrays = len(lags)
        
        for i in range(self.n_feat_arrays):
            self.feat_history[i] = []
            
    def append_feat(self, X, max_lag):
        """
        Append the feature vectors in X to feat_history dict
        
        Parameters:
            
            X: features. Either an array of dimension (n_samples, n_features)
               or a list of arrays of dimension (n_samples, n_features)
        """

        #if X is one array, add it to a list anyway
        if type(X) == np.ndarray:
            tmp = []
            tmp.append(X)
            X = tmp
        
        for i in range(self.n_feat_arrays):
            self.feat_history[i].append(X[i])
                        
            #if max number of features is reached, remove first item
            if len(self.feat_history[i]) > max_lag:
                self.feat_history[i].pop(0)
                
    def get_feat_history(self):
        """
        Return the features from the feat_history dict based on the lags
        specified in self.lags
        
        Returns:
            X_i: array of lagged features of dimension (feat1.size + feat2.size + ...,) 
        """
        X_i = []
        
        idx = 0
        for i in range(self.n_feat_arrays):
            for lag in self.lags[idx]:
                X_i.append(self.feat_history[i][-lag])
            idx += 1
            
        return np.array(list(chain(*X_i)))