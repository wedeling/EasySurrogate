"""
==============================================================================
CLASS FOR A QUANTIZED SOFTMAX NETWORK
------------------------------------------------------------------------------
Author: W. Edeling
Source: Resampling with neural networks for stochastic parameterization in multiscale systems
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

    ############################
    # START COMMON SUBROUTINES #
    ############################

    def train(self, feats, target, lags, n_iter,
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

        if not isinstance(feats, list):
            feats = [feats]

        self.lags = lags
        self.max_lag = np.max(list(chain(*self.lags)))
        self.n_bins = n_bins

        # Feature engineering object
        self.feat_eng = es.methods.Feature_Engineering()

        # number of training samples
        self.n_samples = feats[0].shape[0]
        # compute the size of the training set based on value of test_frac
        self.n_train = np.int(self.n_samples * (1.0 - test_frac))
        print('Using first %d/%d samples to train QSN' % (self.n_train, self.n_samples))

        # list of features
        X = [X_i[0:self.n_train] for X_i in feats]
        # the data
        y = target[0:self.n_train]

        # True/False on wether the X features are symmetric arrays or not
        if 'X_symmetry' in kwargs:
            self.X_symmetry = kwargs['X_symmetry']
        else:
            self.X_symmetry = np.zeros(len(X), dtype=bool)

        print('Creating time-lagged training data...')
        X_train, y_train = self.feat_eng.lag_training_data(X, y, lags=lags,
                                                           X_symmetry=self.X_symmetry)
        print('done')

        # number of softmax layers (one per output)
        n_softmax = y_train.shape[1]

        # number of output neurons
        n_out = n_bins * n_softmax

        # one-hot encoded y data e.g. [0, 0, 1, 0, ..., 0] if the y sample
        # falls in the 3rd bin
        one_hot_encoded_data = self.feat_eng.bin_data(y_train, n_bins)

        # simple sampler to draw random samples from the bins
        self.sampler = es.methods.SimpleBin(self.feat_eng)

        # create the feed-forward QSN
        self.surrogate = es.methods.ANN(X=X_train, y=one_hot_encoded_data,
                                        n_layers=n_layers, n_neurons=n_neurons,
                                        n_softmax=n_softmax, n_out=n_out,
                                        loss='cross_entropy',
                                        activation=activation, batch_size=batch_size,
                                        lamb=lamb, decay_step=10**4, decay_rate=0.9,
                                        standardize_X=True, standardize_y=False,
                                        save=False)

        print('===============================')
        print('Training Quantized Softmax Network...')

        # train network for N_iter mini batches
        self.surrogate.train(n_iter, store_loss=True)
        self.set_feature_stats()
        self.init_feature_history(feats)

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
        if not isinstance(X, list):
            X = [X]
        # append the current state X to the feature history
        self.feat_eng.append_feat(X)
        # get the feature history defined by the specified number of time lags.
        # Here, feat is an array with the same size as the neural network input layer
        feat = self.feat_eng.get_feat_history()
        # features were standardized during training, do so here as well
        feat = (feat - self.feat_mean) / self.feat_std
        # o_i = the probability mass function at the output layer
        # max_idx = the bin index with the highest probability
        o_i, max_idx, _ = self.surrogate.get_softmax(feat.reshape([1, self.surrogate.n_in]))
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

    def set_feature_stats(self):
        """
        If the features were standardized, this stores the mean and
        standard deviation of the features in the QSN_Surrogate object.
        """
        if hasattr(self.surrogate, 'X_mean'):
            self.feat_mean = self.surrogate.X_mean
            self.feat_std = self.surrogate.X_std

    def init_feature_history(self, feats, start=0):
        """
        The features are assumed to be lagged in time. Therefore, the initial
        time-lagged feature vector must be set up. The training data is used
        for this.

        Parameters
        ----------
        + feats : a list of the variables used to construct the time-lagged
        features: [var_1, var_2, ...]. Each var_i is an array such that
        var_i[0] gives the value of var_i at t_0, var_i[1] at t_1 etc.

        + start : the starting index of the training features. Default is 0.

        Returns
        -------
        None.

        """
        for i in range(self.max_lag):
            feat = [X_i[start + i] for X_i in feats]
            self.feat_eng.append_feat(feat)
