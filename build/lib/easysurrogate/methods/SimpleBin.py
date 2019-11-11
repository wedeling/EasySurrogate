import numpy as np
from scipy import stats
from itertools import chain

class SimpleBin:
    
    def __init__(self, y, n_bins):

        n_samples = y.shape[0]
        
        if y.ndim == 2:
            n_vars = y.shape[1]
        else:
            n_vars = 1
            y = y.reshape([n_samples, 1])
        
        self.n_vars = n_vars
        self.binnumbers = np.zeros([n_samples, n_vars]).astype('int')
        self.r_ip1 = {}
        self.bins = {}
        self.bin_data = np.zeros([n_samples, n_bins*n_vars])
        
        for i in range(n_vars):
            
            self.r_ip1[i] = {}
                       
            bins = np.linspace(np.min(y[:, i]), np.max(y[:, i]), n_bins+1)
            self.bins[i] = bins

            count, _, self.binnumbers[:, i] = \
            stats.binned_statistic(y[:, i], np.zeros(n_samples), statistic='count', bins=bins)
        
            unique_binnumbers = np.unique(self.binnumbers[:, i])

            offset = i*n_bins

            for j in unique_binnumbers:
                idx = np.where(self.binnumbers[:, i] == j)
                self.r_ip1[i][j-1] = y[idx, i]
                
                self.bin_data[idx, offset + j - 1] = 1.0
            
#        self.mapping = np.zeros(self.n_bins + 2).astype('int')
#        self.mapping[1:-1] = range(self.n_bins)
#        self.mapping[-1] = self.n_bins - 1
            
    def draw(self, bin_idx):
        
        pred = np.zeros(self.n_vars)
        
        for i in range(self.n_vars):
            pred[i] = np.random.choice(self.r_ip1[i][bin_idx[i]][0])
        
        return pred
    
#    #append the features supplied to the binning object during simulation to self.feat
#    #Note: use list to dynamically append, array is very slow
#    #Note 2: X_i must be of size [n_in, 1], where n_in is 
#    #the number of input features at a SINGLE time instance
#    def append_feat(self, X_i):
#       
#        for i in range(self.n_feat):
#            self.feat[i].append(X_i[i])
#            
#            #if max number of lagged features is reached, remove first item
#            if len(self.feat[i]) > self.max_lag:
#                self.feat[i].pop(0)
#
#    #return all time lagged features in self.feat as a 1d array
#    def get_all_feats(self):
#        return np.array(list(chain(*self.feat.values())))