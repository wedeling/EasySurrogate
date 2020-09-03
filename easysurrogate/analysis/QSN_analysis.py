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

    def __init__(self, **kwargs):
        print('Creating QSN_analysis object')

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