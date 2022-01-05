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


class GP_Surrogate(Campaign):

    def __init__(self, backend='scikit-learn', **kwargs):
        """
        GP_surrogate class for Gaussian Process Regression

        """
        print('Creating Gaussian Process Object')

        self.name = 'GP Surrogate'
        self.feat_eng = es.methods.Feature_Engineering()
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.backend = backend

        if 'noise' in kwargs:
            self.noise = kwargs['noise']

        if 'n_in' in kwargs:
            self.n_in = kwargs['n_in']
        else:
            self.n_in = 1

        if 'n_out' in kwargs:
            self.n_out = kwargs['n_out']
        else:
            self.n_out = 1

    ############################
    # START COMMON SUBROUTINES #
    ############################

    def train(self, feats, target, n_iter=0,
              test_frac=0.0,
              **kwargs):
        """

        Args:
            feats: feature array, or list of different feature arrays
            target: the target data
            n_iter: number of hyperoptimisation restarts
            test_frac: Fraction of the data used for training

        Returns:
        -------
        None.
        """

        if 'basekernel' not in kwargs:
            self.base_kernel = 'Matern'
        else:
            self.base_kernel = kwargs['basekernel']

        if 'noize' not in kwargs:
            self.noize = 'True'
        else:
            self.noize = kwargs['noize']

        # prepare the training data
        X_train, y_train, X_test, y_test = self.feat_eng.get_training_data(
            feats, target, local=False, test_frac=test_frac, train_first=False)

        # scale the training data
        X_train = self.x_scaler.fit_transform(X_train)
        y_train = self.y_scaler.fit_transform(y_train)
        if len(X_test) > 0 and len(y_test) > 0:
            X_test = self.x_scaler.transform(X_test)
            y_test = self.y_scaler.transform(y_test)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # create a GP process
        print('===============================')
        print('Fitting Gaussian Process...')
        self.model = es.methods.GP(
            kernel=self.base_kernel,
            n_in=self.n_in,
            n_out=self.n_out,
            bias=False,
            noize=self.noize,
            backend=self.backend)

        # get dimensionality of the output
        self.n_out = y_train.shape[1]

        # get the dimesionality of te input
        self.n_in = X_train.shape[1]

        self.model.train(self.X_train, self.y_train)

    def predict(self, X):
        """
        Make a stochastic prediction of the output y conditional on the
        input features [X]

        Args:
            X: list of feature arrays, the state given at the point

        Returns:
        -------
        Stochastic prediction of the output y
        """
        # TODO slows down a lot, maybe FeatureEngineering should return training data still as list
        x = np.array([x for x in X]).T
        x = self.x_scaler.transform(x)
        x = [np.array(i) for i in x.T.tolist()]

        # TODO unlike ANNs, GPs should provide API for vectorised .predict() and other methods
        y, std, _ = self.feat_eng._predict(x, feed_forward=lambda t: self.model.predict(t))

        y = self.y_scaler.inverse_transform(y)

        self.y_scaler.with_mean = False
        std = self.y_scaler.inverse_transform(std * np.ones(y.shape))
        self.y_scaler.with_mean = True

        return y, std

    def save_state(self, state=None, **kwargs):
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
            self.feat_mean = 0.0 * np.ones((1, self.n_in))
            self.feat_std = 1.0 * np.ones((1, self.n_in))

        if hasattr(self.model, 'y_mean'):
            self.output_mean = self.model.y_mean
            self.output_std = self.model.y_std
        else:
            self.output_mean = 0.0 * np.ones((1, self.n_out))
            self.output_std = 1.0 * np.ones((1, self.n_out))

    def train_sequentially(self, feats=None, target=None,
                           n_iter=0, **kwargs):
        """
        Update GP surrogate with a sequeantial design scheme

        Parameters
        ----------
        feats: list of feature arrays
        target: array of target data
        n_iter: integer, number of iterations of sequential optimisation
        """

        self.set_data_stats()

        if 'acquisition_function' in kwargs:
            acq_func_arg = kwargs['acquisition_function']
            if acq_func_arg == 'poi':
                acq_func_obj = self.poi_acquisition_function
            elif acq_func_arg == 'mu':
                acq_func_obj = self.maxunc_acquisition_function
            else:
                raise NotImplementedError(
                    'This rule for sequential optimisation is not implemented, using default.')
        else:
            acq_func_obj = self.maxunc_acquisition_function

        if 'save_history' in kwargs:
            save_history = kwargs['save_history']
        else:
            save_history = False

        if save_history:
            self.design_history = []

        if self.backend == 'scikit-learn':

            """
            0) iterate for n_iter
                1) state on step n: object has X_train, X_test, their indices, model instance
                2) find set of candidates at minima of acq function X_cand; now object has
                X_train:=X_train U X_cand, X_test = X_test U_ X_cand, global inidces and set sizes updated
                3) model instance is updated : first approach to train new model for new X_train
            """

            for i in range(n_iter):

                X_new, x_new_ind_test, x_new_ind_glob = self.feat_eng.\
                    chose_feature_from_acquisition(acq_func_obj, self.X_test)
                X_new = X_new.reshape(1, -1)

                # x_new_inds = feats.index(X_new)  # feats is list of features, for this
                # has to be list of samples
                y_new = self.y_test[x_new_ind_test].reshape(1, -1)

                self.feat_eng.train_indices = np.concatenate([self.feat_eng.train_indices,
                                                              np.array(x_new_ind_glob).reshape(-1)])
                self.feat_eng.test_indices = np.delete(
                    self.feat_eng.test_indices, x_new_ind_test, 0)
                self.feat_eng.n_train += 1
                self.feat_eng.n_test -= 1

                X_train = np.concatenate([self.X_train, X_new])
                y_train = np.concatenate([self.y_train, y_new])
                X_test = np.delete(self.X_test, x_new_ind_test, 0)
                y_test = np.delete(self.y_test, x_new_ind_test, 0)

                if save_history:
                    self.design_history.append(x_new_ind_test)

                # TODO update the scaler - has to transform back to original values, then refit
                #X_train = self.x_scaler.fit_transform(X_train)
                #y_train = self.y_scaler.fit_transform(y_train)
                #X_test = self.x_scaler.transform(X_test)
                #y_test = self.y_scaler.transform(y_test)

                self.X_train = X_train
                self.y_train = y_train
                self.X_test = X_test
                self.y_test = y_test

                # self.model = es.methods.GP(X_train, y_train,
                # kernel=self.base_kernel, bias=False, noize=self.noize,
                # backend=self.backend)

                self.model.train(X_train, y_train)

        elif self.backend == 'mogp':
            pass
        else:
            raise NotImplementedError('Currently supporting only scikit-learn and mogp backend')

    def derivative_x(self, X):
        """
        Make a prediction of the derivative of output y by input x

        Args:
            X: list of feature arrays

        Returns:
        -------
        dy/dx
        """
        # TODO ideally should be a jacobian (dy_i/dx_j)_ij
        if self.backend == 'mogp':

            x = np.array([x for x in X]).T
            x = self.x_scaler.transform(x)
            x = [np.array(i) for i in x.T.tolist()]

            _, _, der = self.feat_eng._predict(x, feed_forward=lambda t: self.model.predict(t))
        else:
            raise NotImplementedError(
                "Gaussian Process derivatives w.r.t. inputs are implemented only for MOGP")

        self.y_scaler.with_mean = False
        der = self.y_scaler.inverse_transform(der)
        self.y_scaler.with_mean = True

        self.x_scaler.with_mean = False
        der = self.x_scaler.transform(der)
        self.x_scaler.with_mean = True

        return der

    def maxunc_acquisition_function(self, sample, candidates=None):
        """
        Returns the uncertainty of the model as (a posterior variance on Y) for a given sample
        Args:
            sample: a single sample from a feature array
            candidates: list of input parameter files to chose optimum from
        Returns:
            the value of uncertainty (variance) of the model
        """

        if sample.ndim == 1:
            sample = sample[None, :]

        _, uncertatinty, _ = self.model.predict(sample)

        return -1. * uncertatinty

    def poi_acquisition_function(self, sample, candidates=None):
        """
        Returns the probability of improvement for a given sample
        Args:
            sample: a single sample from a feature array
            candidates: list of input parameter files to chose optimum from
        Returns:
            the probability of improvement if a given sample will be added to the model
        """

        jitter = 1e-9
        f_star = self.output_mean

        if sample.ndim == 1:
            sample = sample[None, :]

        mu, std, d = self.model.predict(sample)
        poi = np.linalg.norm(np.divide(abs(mu - f_star), std + jitter), ord=2)

        return -poi
