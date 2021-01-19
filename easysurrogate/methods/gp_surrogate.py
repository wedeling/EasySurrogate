"""
CLASS FOR A GAUSSIAN PROCESS REGRESSION
------------------------------------------------------------------------------
Author: Y. Yudin
==============================================================================
"""

import numpy as np
from ..campaign import Campaign
import easysurrogate as es

class GP_Surrogate(Campaign):

    def __init__(self, **kwargs):
        print('Creating Gaussain Process Object')
        self.name = 'GP Surrogate'
        self.feat_eng = es.methods.Feature_Engineering()


    ############################
    # START COMMON SUBROUTINES #
    ############################

    def train(self, feats, target, n_iter=0,
              test_frac=0.0,
              kernel=):
        """

        Args:
            feats: feature array, or list of different feature arrays
            target: the target data
            n_iter:
            test_frac: Fraction of the data used for training
            kernel: type of covariance function kernel used for gaussian process

        Returns:
        -------
        None.
        """

        # prepare the training data
        X_train, y_train = self.feat_eng.get_training_data(feats, target, test_frac=test_frac)

        # get dimensionality of the output
        n_out = y_train.shape[1]

        # create a GP process
        self.model = es.methods.GP()

        print('===============================')
        print('Fitting Gaussian Process...')


    def predict(self, X):
        """
        Make a stochastic prediction of the output y conditional on the
        input features [X]

        Args:
            X: the state given at the given point

        Returns:
        -------
        Stochastic prediction of the output y
        """
        return self.feat_eng._predict(X, function=lambda x: x)

    def save_state(self):
        """
        Save the state of GP surrogate as a pickle file
        """
        state = self.__dict__
        super().save_state(state=state, name=self.name)

    def load_state(self):
        """
        Load the state of the GP surrogate from file
        """
        super.load_state(name=self.name)

    ##########################
    # END COMMON SUBROUTINES #
    ##########################

    def set_data_stats(self):
        """
        If the data were standardized, this stores the mean and
        standard derivation of the features in the GP_Surrogate object.
        """
        if hasattr(self.model, 'X_mean'):
            self.feat_mean = self.model.X_mean
            self.seat_Std = self.model.X_std
        else:
            self.feat_mean = 0.0
            self.feat_std = 1.0

        if hasattr(self.model, 'y_mean'):
            self.output_mean = self.model.y_mean
            self.output_std = self.model.y_std
        else:
            self.output_mean = 0.0
            self.output_std = 1.0
