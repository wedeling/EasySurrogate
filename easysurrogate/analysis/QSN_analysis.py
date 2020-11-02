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

    def get_classification_error(self, index=None, **kwargs):
        """
        Compute the misclassification error of the QSN surrogate.

        Parameters
        ----------

        index : array, default is None
            indices to select a subset of feature/data points to perform test on. When None
            the classification error is computed on the entire dataset.

        Returns
        -------
        Prints classification error to screen for every softmax layer

        """

        if index is not None:
            X = self.qsn_surrogate.surrogate.X[index]
            y = self.qsn_surrogate.surrogate.y[index]
        else:
            X = self.qsn_surrogate.surrogate.X
            y = self.qsn_surrogate.surrogate.y

        self.qsn_surrogate.surrogate.compute_misclass_softmax(X=X, y=y)
