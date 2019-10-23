import numpy as np
from sklearn.neighbors.kde import KernelDensity
import tkinter as tk
from tkinter import filedialog
import h5py

"""
===============================================================================
CLASS FOR POST PROCESSING SUBROUTINES
===============================================================================
"""

class Post_Processing:
    
    def __init__(self):
        """
        """
        
    def auto_correlation_function(self, X, max_lag):
        """
        Compute the autocorrelation of X over max_lag time steps
        
        Parameters:
            - X (array): the samples from which to compute the ACF 
            - max_lag (int): the max number of time steps, determines max 
              lead time
              
        Returns:
            - R (array): array of ACF values
        """
        lags = np.arange(1, max_lag)
        R = np.zeros(lags.size)
        
        idx = 0
        
        #for every lag, compute autocorrelation:
        # R = E[(X_t - mu_t)*(X_s - mu_s)]/(std_t*std_s)
        for lag in lags:
        
            X_t = X[0:-lag]
            X_s = X[lag:]
        
            mu_t = np.mean(X_t)
            std_t = np.std(X_t)
            mu_s = np.mean(X_s)
            std_s = np.std(X_s)
        
            R[idx] = np.mean((X_t - mu_t)*(X_s - mu_s))/(std_t*std_s)
            idx += 1
    
        return R
    
    def get_pde(self, X, Npoints = 100):
        """
        Computes a kernel density estimate of the samples in X   
        
        Parameters:
            - X (array): the samples
            - Npoints (int, default = 100): the number of points in the domain of X
            
        Returns:
            - domain (array of size (Npoints,): the domain of X
            - kde (array of size (Npoints,): the kernel-density estimate
        """
 
        #    kernel = stats.gaussian_kde(X, bw_method='scott')
        #    x = np.linspace(np.min(X), np.max(X), Npoints)
        #    pde = kernel.evaluate(x)
        #    return x, pde
        
        print('Computing kernel-density estimate')
            
        X_min = np.min(X)
        X_max = np.max(X)
        bandwidth = (X_max-X_min)/40
        
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X.reshape(-1, 1))
        domain = np.linspace(X_min, X_max, Npoints).reshape(-1, 1)
        log_dens = kde.score_samples(domain)
        
        print('done')
        
        return domain, np.exp(log_dens)
    
    def store_samples_hdf5(samples, names = [], file_path = ""):
        """
        store samples in hierarchical data format
        """
        
        #if X is one array, add it to a list anyway
        if type(samples) == np.ndarray:
            tmp = []
            tmp.append(samples)
            samples = tmp
            
        if names == []:
            for i in len(samples):
                names.append('sample_' + str(i))
        
        if len(file_path) == 0:

            root = tk.Tk()
            root.withdraw()
            
            file = filedialog.asksaveasfile(mode='wb', defaultextension=".pickle")
        else:
            file = open(file_path, 'wb')

        print('Saving samples in', file.name) 
        
        #create HDF5 file
        h5f = h5py.File(file, 'w')
        
        #store numpy sample arrays as individual datasets in the hdf5 file
        if type(samples) == list:
            idx = 0
            for name in names:
                h5f.create_dataset(name, data = samples[idx])
                idx += 1
        elif type(samples) == dict:
            for name in samples.keys():
                h5f.create_dataset(name, data = samples[name])
        else:
            print("Error: samples must be a dict or a list")
            return

        h5f.close()
        print('done')