"""
CLASS TO PERFORM ANALYSIS ON RESULTS FROM A QUANTIZED SOFTMAX SURROGATE.
"""
from .base import BaseAnalysis
import numpy as np
# from sklearn.neighbors import KernelDensity

class QSN_analysis(BaseAnalysis):
    """
    QSN analysis class
    """

    def __init__(self, qsn_surrogate, **kwargs):
        print('Creating QSN_analysis object')
        self.qsn_surrogate = qsn_surrogate

    def auto_correlation_function(self, X, max_lag):
        """
        Compute the autocorrelation of X over max_lag time steps

        Parameters:
            - X (array, size (N,)): the samples from which to compute the ACF 
            - max_lag (int): the max number of time steps, determines max 
              lead time

        Returns:
            - R (array): array of ACF values
        """
        return super().auto_correlation_function(X, max_lag)

    def cross_correlation_function(self, X, Y, max_lag):
        """
        Compute the crosscorrelation between X and Y over max_lag time steps
        
        Parameters:
            - X, Y (array, size (N,)): the samples from which to compute the CCF 
            - max_lag (int): the max number of time steps, determines max 
              lead time

        Returns:
            - C (array): array of CCF values
        """
        return super().cross_correlation_function(X, Y, max_lag)

    def get_pdf(self, X, Npoints = 100):
        """
        Computes a kernel density estimate of the samples in X   
        
        Parameters:
            - X (array): the samples
            - Npoints (int, default = 100): the number of points in the domain of X
            
        Returns:
            - domain (array of size (Npoints,): the domain of X
            - kde (array of size (Npoints,): the kernel-density estimate
        """
        return super().get_pdf(X, Npoints=Npoints)

    def get_classification_error(self, features, target, **kwargs):
        """
        Compute the misclassification error of the QSN surrogate.

        Parameters
        ----------
        + features : an array of multiple input features.
        + y : an array of multiple target data points

        Returns
        -------
        Prints classification error to screen for every softmax layer

        """

        if not type(features) is list: features = [features]

        #True/False on wether the X features are symmetric arrays or not
        if 'X_symmetry' in kwargs:
            X_symmetry = kwargs['X_symmetry']
        else:
            X_symmetry = np.zeros(len(features), dtype=bool)

        print('Creating time-lagged training data...')
        X, y = self.qsn_surrogate.feat_eng.lag_training_data(features, target,
                                                             lags=self.qsn_surrogate.lags, 
                                                             X_symmetry=X_symmetry)
        #create one-hot encoded training data per y sample
        one_hot_encoded_data = self.qsn_surrogate.feat_eng.bin_data(y, self.qsn_surrogate.n_bins)

        self.qsn_surrogate.surrogate.compute_misclass_softmax(X = X, y = one_hot_encoded_data)