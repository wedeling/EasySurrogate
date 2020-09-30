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

    def get_classification_error(self, features, targets, index=None, **kwargs):
        """
        Compute the misclassification error of the QSN surrogate.

        Parameters
        ----------
        features : list of array
            a list of multiple feature arrays or a single feature array

        y : array
            an array of target data points

        index : array, default is None
            indices to select a subset of feature/data points to perform test on. When None
            the classification error is computed on the entire dataset.

        Returns
        -------
        Prints classification error to screen for every softmax layer

        """

        if not isinstance(features, list):
            features = [features]

        if (not isinstance(index, None)) and (not isinstance(index, np.ndarray)):
            print('QSNAnalysis.get_classification_error: index argument must be None or an array')
            return

        # True/False on wether the X features are symmetric arrays or not
        if 'X_symmetry' in kwargs:
            X_symmetry = kwargs['X_symmetry']
        else:
            X_symmetry = np.zeros(len(features), dtype=bool)

        # select a subset of the data if provided
        if index is not None:
            features = [feature[index] for feature in features]
            targets = targets[index]

        print('Creating time-lagged training data...')
        X, y = self.qsn_surrogate.feat_eng.lag_training_data(features, targets,
                                                             lags=self.qsn_surrogate.lags,
                                                             X_symmetry=X_symmetry)

        if hasattr(self.qsn_surrogate, 'feat_mean'):
            X = (X - self.qsn_surrogate.feat_mean) / self.qsn_surrogate.feat_std

        # create one-hot encoded training data per y sample
        one_hot_encoded_data = self.qsn_surrogate.feat_eng.bin_data(y, self.qsn_surrogate.n_bins)

        self.qsn_surrogate.surrogate.compute_misclass_softmax(X=X, y=one_hot_encoded_data)
