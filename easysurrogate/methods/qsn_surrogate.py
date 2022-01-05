"""
==============================================================================
CLASS FOR A QUANTIZED SOFTMAX NETWORK
------------------------------------------------------------------------------
Author: W. Edeling
Source: Resampling with neural networks for stochastic parameterization in
        multiscale systems
arXiv preprint arXiv:2004.01457
==============================================================================
"""
from itertools import chain
import numpy as np
# import h5py
from ..campaign import Campaign
import easysurrogate as es


class QSN_Surrogate(Campaign):

    def __init__(self, **kwargs):
        print('Creating Quantized Softmax Object')
        self.name = 'QSN Surrogate'
        # create a Feature_Engineering object
        self.feat_eng = es.methods.Feature_Engineering()

    ############################
    # START COMMON SUBROUTINES #
    ############################

    def train(self, feats, target, n_iter, lags=None, local=False,
              n_bins=10, test_frac=0.0,
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
        n_iter : number of back propagation iterations
        n_bins : Number of output bins. The default is 10.
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

        # is a single array is provided, also put it in a list
        if isinstance(feats, np.ndarray):
            feats = [feats]

        # time lags
        self.lags = lags

        self.n_bins = n_bins

        # flag if the surrogate is to be applied locally or not
        self.local = local

        # test fraction
        self.test_frac = test_frac

        # prepare the training data
        X_train, y_train, _, _ = self.feat_eng.get_training_data(
            feats, target, lags=lags, local=local, test_frac=test_frac)
        # get the maximum lag that was specified
        self.max_lag = self.feat_eng.max_lag

        # number of softmax layers (one per output)
        self.n_softmax = y_train.shape[1]

        # number of output neurons
        n_out = n_bins * self.n_softmax

        # one-hot encoded y data e.g. [0, 0, 1, 0, ..., 0] if the y sample
        # falls in the 3rd bin
        one_hot_encoded_data = self.feat_eng.bin_data(y_train, n_bins)

        # simple sampler to draw random samples from the bins
        self.sampler = es.methods.SimpleBin(self.feat_eng)

        # create the feed-forward QSN
        self.neural_net = es.methods.ANN(X=X_train, y=one_hot_encoded_data,
                                         n_layers=n_layers, n_neurons=n_neurons,
                                         n_softmax=self.n_softmax, n_out=n_out,
                                         loss='cross_entropy',
                                         activation=activation, batch_size=batch_size,
                                         lamb=lamb, decay_step=10**4, decay_rate=0.9,
                                         standardize_X=True, standardize_y=False,
                                         save=False)

        print('===============================')
        print('Training Quantized Softmax Network...')

        # train network for N_iter mini batches
        self.neural_net.train(n_iter, store_loss=True)
        self.set_data_stats()
        if lags is not None:
            self.feat_eng.initial_condition_feature_history(feats)

    def predict(self, X):
        """
        Make a stochastic prediction of the output y conditional on the
        input features [X_t, X_{t-lag1}, X_{t-lag2}, ...]

        Parameters
        ----------
        X: the state at the current (time) step. If the QSN is conditioned on
        more than 1 (time-lagged) variable, X must be a list containing all
        variables at the current time step.

        Returns
        -------
        Stochastic prediction of the output y

        """

        return self.feat_eng._predict(X, self._feed_forward)

    def _feed_forward(self, feat):
        """

        A feed forward run of the QSN. This is the only part of prediction that is specific
        to QSN_Surrogate.

        Parameters
        ----------
        feat : array of list of arrays
               The feature array of a list of feature arrays on which to evaluate the surrogate.

        Returns
        -------
        y : array
            the stochastic prediction of the QSN.
        """

        # features were standardized during training, do so here as well
        feat = (feat - self.feat_mean) / self.feat_std
        # o_i = the probability mass function at the output layer
        # max_idx = the bin index with the highest probability
        o_i, max_idx, _ = self.neural_net.get_softmax(feat.reshape([1, self.neural_net.n_in]))
        # resample a value from the selected bin
        return self.sampler.resample(max_idx)

    def save_state(self):
        """
        Save the state of the QSN surrogate to a pickle file
        """
        state = self.__dict__
        super().save_state(state=state, name=self.name)

    def load_state(self):
        """
        Load the state of the QSN surrogate from file
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
