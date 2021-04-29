"""
CLASS FOR A GAUSSIAN PROCESS REGRESSION
------------------------------------------------------------------------------
Author: Y. Yudin
==============================================================================
"""

import numpy as np
from ..campaign import Campaign
import easysurrogate as es

from sklearn.preprocessing import StandardScaler

#DEBUG
from matplotlib import pyplot as plt

class GP_Surrogate(Campaign):

    def __init__(self, backend='scikit-learn', **kwargs):
        print('Creating Gaussain Process Object')
        self.name = 'GP Surrogate'
        self.feat_eng = es.methods.Feature_Engineering()
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.backend = backend

        if 'noise' in kwargs:
            self.noise = kwargs['noise']

    ############################
    # START COMMON SUBROUTINES #
    ############################

    def train(self, feats, target, n_iter=0,
              test_frac=0.0, postrain=False,
              **kwargs):
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

        if 'base_kernel' not in kwargs:
            base_kernel = 'Matern'
        else:
            base_kernel = kwargs['basekernel']

        if 'noize' not in kwargs:
            noize = 'True'
        else:
            noize = kwargs['noize']

        # prepare the training data
        if not postrain:
            X_train, y_train, X_test, y_test = self.feat_eng.get_training_data(feats, target,
                                                            local=False, test_frac=test_frac, train_first=True)

            # scale the training data
            X_train = self.x_scaler.fit_transform(X_train)
            y_train = self.y_scaler.fit_transform(y_train)
            X_test = self.x_scaler.transform(X_test)
            y_test = self.y_scaler.transform(y_test)

            # create a GP process
            self.model = es.methods.GP(X_train, y_train,
                                       kernel=base_kernel, bias=False, noize=noize, backend=self.backend)

        elif postrain:
            X_train, y_train, X_test, y_test = \
                                self.feat_eng.get_training_data(feats, target, local=False, test_frac=test_frac,
                                        train_sample_choice=lambda x: self.acquisition_function(x))  # case with acquisition

            X = feats[self.feat_eng.train_indices]
            y = target[self.feat_eng.train_indices]

            X = np.concatenate([X, X_train])
            y = np.concatenate([y, y_train])
            self.model.train(X, y)

        # get dimensionality of the output
        self.n_out = y_train.shape[1]

        print('===============================')
        print('Fitting Gaussian Process...')

        # # DEBUG
        # print(feats)
        # print(X_test)
        # y_pred = self.model.instance.predict(X_test)
        # err = abs(np.divide(y_pred - y_test, y_test))
        # print(err)
        # y_pred_tr = self.model.instance.predict(X_train)
        # err_tr = np.divide(abs(y_pred_tr - y_train), y_train)
        # print(err_tr)
        # plt.plot()

    def predict(self, X):
        """
        Make a stochastic prediction of the output y conditional on the
        input features [X]

        Args:
            X: list of feture arrays, the state given at the point

        Returns:
        -------
        Stochastic prediction of the output y
        """
        x = np.array([x.reshape(-1) for x in X]).T  # TODO slows down a lot, maybe FeatureEngineering should return training data still as list
        x = self.x_scaler.transform(x)
        x = [np.array(i).reshape(-1, 1) for i in x.T.tolist()]
        y, std = self.feat_eng._predict(X, feed_forward=lambda x: self.model.predict(x))  #TODO check if there is a way to pass right shape of sample

        #return self.model.predict(X)

        y = self.y_scaler.inverse_transform(y)
        std = self.y_scaler.inverse_transform(std)

        return y, std

    def save_state(self):
        """
        Save the state of GP surrogate as a pickle file
        """
        state = self.__dict__
        save_state = super().save_state(state=state, name=self.name)

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

    def maxunc_acquisition_function(self, sample):
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

    def poi_acquisition_function(self, sample):
        """
        Returns the probability of improvement for a given sample
        Args:
            sample: a single sample from a feature array
        Returns:
            the probability of improvement if a given sample will be added to the model
        """
        jitter = 1e-9
        f_star = self.output_mean

        if sample.ndim == 1:
            sample = sample[None, :]

        mu, std = self.model.predict(sample)
        poi = np.divide(mu - f_star, std + jitter)

        return poi

