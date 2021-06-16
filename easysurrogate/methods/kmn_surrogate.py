"""
==============================================================================
CLASS FOR A KERNEL MIXTURE NETWORK
------------------------------------------------------------------------------
Author: W. Edeling
Source: Luca Ambrogioni et al, The Kernel Mixture Network:
        A Nonparametric Method for Conditional Density Estimation of
        Continuous Random Variables, 2017.
==============================================================================
"""

from itertools import chain, product
import numpy as np
from scipy.stats import norm
from ..campaign import Campaign
import easysurrogate as es


class KMN_Surrogate(Campaign):

    def __init__(self, **kwargs):
        print('Creating Kernel Mixture Network Object')
        self.name = 'KMN Surrogate'
        # create a Feature_Engineering object
        self.feat_eng = es.methods.Feature_Engineering()

    ############################
    # START COMMON SUBROUTINES #
    ############################

    def train(self, feats, target, n_iter,
              kernel_means, kernel_stds, n_softmax,
              lags=None, local=False,
              test_frac=0.0,
              n_layers=2, n_neurons=100,
              activation='leaky_relu',
              batch_size=64, lamb=0.0, **kwargs):
        """
        Perform back propagation to train the QSN

        Parameters
        ----------
        feats : feature array, or list of different feature arrays
        target : the target data
        lags : list of time lags per feature
        local : apply the surrogate locally at each grid point.
        n_iter : number of back propagation iterations
        kernel_means : achor points of the kernels
        kernel_stds : standard deviations of the kernels
        test_frac : Fraction of the data to use for training.
                    The default is 0.0.
        n_layers : The number of layers in the neural network. Includes the
                   input layer and the hidden layers. The default is 2.
        n_neurons : The number of neurons per layer. The default is 100.
        activation : Type of activation function. The default is 'leaky_relu'.
        batch_size : Mini batch size. The default is 64.
        lamb : L2 regularization parameter. The default is 0.0.

        Returns
        -------
        None.

        """

        self.lags = lags
        self.local = local

        # number of softmax layers (one per output)
        self.n_softmax = n_softmax

        # create all possible combinations of the specified kernel means and std devs
        self.kernel_means = []
        self.kernel_stds = []
        for i in range(self.n_softmax):
            combi = np.array(list(chain(product(kernel_means[i], kernel_stds[i]))))
            self.kernel_means.append(combi[:, 0].reshape([-1, 1]))
            self.kernel_stds.append(combi[:, 1].reshape([-1, 1]))

        # size of a single softmax layer
        self.n_bins = self.kernel_means[0].size

        # prepare the training data
        X_train, y_train, _, _ = self.feat_eng.get_training_data(feats,
                                                                 target,
                                                                 lags=lags,
                                                                 local=local,
                                                                 test_frac=test_frac)

        # get the maximum lag that was specified
        self.max_lag = self.feat_eng.max_lag

        # number of output neurons
        n_out = self.n_bins * self.n_softmax

        # create the feed-forward QSN
        self.neural_net = es.methods.ANN(X=X_train, y=y_train,
                                         n_layers=n_layers, n_neurons=n_neurons,
                                         n_softmax=self.n_softmax, n_out=n_out,
                                         loss='kernel_mixture',
                                         activation=activation, batch_size=batch_size,
                                         lamb=lamb, decay_step=10**4, decay_rate=0.9,
                                         standardize_X=True, standardize_y=False,
                                         save=False,
                                         kernel_means=self.kernel_means,
                                         kernel_stds=self.kernel_stds)

        print('===============================')
        print('Training Kernel Mixture Network...')

        # train network for N_iter mini batches
        self.neural_net.train(n_iter, store_loss=True)
        self.set_data_stats()
        if lags is not None:
            self.feat_eng.initial_condition_feature_history(feats)
        # flatten the kernel properties into a single vector (size=#output neurons),
        # used in predict subroutine
        self.kernel_means_flat = np.concatenate(self.kernel_means)
        self.kernel_stds_flat = np.concatenate(self.kernel_stds)

    def predict(self, X):
        """
        Make a stochastic prediction of the output y conditional on the
        input features [X_t, X_{t-lag1}, X_{t-lag2}, ...]

        Parameters
        ----------
        X: the state at the current (time) step. If the KMN is conditioned on
        more than 1 (time-lagged) variable, X must be a list containing all
        variables at the current time step.

        Returns
        -------
        Stochastic prediction of the output y

        """
        # feat_eng._predict handles the preparation of the features and returns
        # self._feed_forward(X)
        return self.feat_eng._predict(X, self._feed_forward)

    def _feed_forward(self, feat):

        feat = (feat - self.feat_mean) / self.feat_std
        # o_i = the probability mass function at the output layer
        # max_idx = the bin index with the highest probability
        o_i, max_idx, _ = self.neural_net.get_softmax(feat.reshape([1, self.neural_net.n_in]))
        self.o_i = o_i
        self.max_idx = max_idx
        # return random sample from the conditional kernel density estimate
        # TODO: implement rvs from softmax layer
        return norm.rvs(self.kernel_means_flat[max_idx], self.kernel_stds_flat[max_idx]).flatten()

    def save_state(self):
        """
        Save the state of the KMN surrogate to a pickle file
        """
        state = self.__dict__
        super().save_state(state=state, name=self.name)

    def load_state(self):
        """
        Load the state of the KMN surrogate from file
        """
        super().load_state(name=self.name)

    ##########################
    # END COMMON SUBROUTINES #
    ##########################

    def set_data_stats(self):
        """
        If the data were standardized, this stores the mean and
        standard deviation of the features in the ANN_Surrogate object.
        """
        if hasattr(self.neural_net, 'X_mean'):
            self.feat_mean = self.neural_net.X_mean
            self.feat_std = self.neural_net.X_std
        else:
            self.feat_mean = 0.0
            self.feat_std = 1.0

        if hasattr(self.neural_net, 'y_mean'):
            self.output_mean = self.neural_net.y_mean
            self.output_std = self.neural_net.y_std
        else:
            self.output_mean = 0.0
            self.output_std = 1.0
