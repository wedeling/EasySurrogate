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
              kernel=['Matern'], postrain=False):
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
        # save all the data
        self.X = feats
        self.y = target

        # prepare the training data
        if postrain == False:
            X_train, y_train, self.X_test, self.y_test = \
                                self.feat_eng.get_training_data(feats, target, local=False, test_frac=test_frac)

            # create a GP process
            self.model = es.methods.GP(X_train, y_train, kernel=kernel[0])

        else:
            X_train, y_train, X_test, y_test = \
                                self.feat_eng.get_training_data(feats, target, local=False, test_frac=test_frac,
                                        train_sample_choice=lambda x: self.acquisition_function(x))  # case with acquisition

            self.model.X = np.concatenate([self.model.X, X_train])
            self.model.y = np.concatenate([self.model.y, y_train])
            self.model.train()

        # get dimensionality of the output
        n_out = y_train.shape[1]

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
        return self.feat_eng._predict(X, feed_forward=lambda x: self.model.predict(x.reshape(1, -1)))  #TODO check if there is a way to pass right shape of sample
        #return self.model.predict(X)

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

    def acquisition_function(self, sample):
        """
        Returns the uncertainty of the model as (a posterior variance on Y) for a given sample
        Args:
            sample: a single sample from a feature array
        Returns:
            the value of uncertainty (variance) of the model
        """
        if sample.ndim == 1:
            sample = sample[None, :]
        _, uncertatinty = self.model.predict(sample)
        return -1.*uncertatinty
