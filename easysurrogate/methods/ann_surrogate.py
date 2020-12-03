"""
==============================================================================
CLASS FOR A ARTIFICIAL NEURAL NETWORK
------------------------------------------------------------------------------
Author: W. Edeling
==============================================================================
"""
from itertools import chain
import numpy as np
# import h5py
from ..campaign import Campaign
import easysurrogate as es


class ANN_Surrogate(Campaign):

    def __init__(self, **kwargs):
        print('Creating ANN_Surrogate Object')
        self.name = 'ANN Surrogate'

    ############################
    # START COMMON SUBROUTINES #
    ############################

    def train(self, feats, target, n_iter, lags=None,
              test_frac=0.0,
              n_layers=2, n_neurons=100,
              activation='tanh',
              batch_size=64, lamb=0.0, **kwargs):
        """
        Perform back propagation to train the ANN

        Parameters
        ----------
        feats : feature array, or list of different feature arrays
        target : the target data
        lags : list of time lags per feature
        n_iter : number of back propagation iterations
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

        # Feature engineering object
        self.feat_eng = es.methods.Feature_Engineering()

        # number of training samples
        self.n_samples = feats[0].shape[0]
        # compute the size of the training set based on value of test_frac
        self.n_train = np.int(self.n_samples * (1.0 - test_frac))
        print('Using first %d/%d samples to train ANN' % (self.n_train, self.n_samples))

        # list of features
        X = [X_i[0:self.n_train] for X_i in feats]
        # the data
        y = target[0:self.n_train]

        # True/False on wether the X features are symmetric arrays or not
        if 'X_symmetry' in kwargs:
            self.X_symmetry = kwargs['X_symmetry']
        else:
            self.X_symmetry = np.zeros(len(X), dtype=bool)

        if lags is not None:
            self.max_lag = np.max(list(chain(*self.lags)))
            print('Creating time-lagged training data...')
            X_train, y_train = self.feat_eng.lag_training_data(X, y, lags=lags,
                                                               X_symmetry=self.X_symmetry)
            print('done')
        else:
            self.max_lag = 0
            X_train = np.array(X).reshape([self.n_train, -1])
            y_train = y

        # number of output neurons
        n_out = y_train.shape[1]

        # create the feed-forward ANN
        self.neural_net = es.methods.ANN(X=X_train, y=y,
                                        n_layers=n_layers, n_neurons=n_neurons,
                                        n_out=n_out,
                                        loss='squared',
                                        activation=activation, batch_size=batch_size,
                                        lamb=lamb, decay_step=10**4, decay_rate=0.9,
                                        standardize_X=True, standardize_y=True,
                                        save=False)

        print('===============================')
        print('Training Artificial Neural Network...')

        # train network for N_iter mini batches
        self.neural_net.train(n_iter, store_loss=True)
        self.set_data_stats()
        if lags is not None:
            self.init_feature_history(feats)

    def train_online(self, feats, target, n_iter,
                     batch_size=1, verbose=False, sequential=True, **kwargs):
        """
        Perform online training, i.e. backpropagation while the surrogate is coupled
        to the macroscopic governing equations.

        Parameters
        ----------
        feats : feature array, or list of different feature arrays
        target : the target data
        n_iter : number of back propagation iterations
        batch_size : Mini batch size. Default is 1. 
        verbose: print loss to screen during back propagation. Default is False.
        sequential: do not randomly sample the training data, Default is True.

        Returns
        -------
        None.

        """

        if not isinstance(feats, list):
            feats = [feats]

        # list of features
        X = [X_i for X_i in feats]

        # True/False on wether the X features are symmetric arrays or not
        if 'X_symmetry' in kwargs:
            self.X_symmetry = kwargs['X_symmetry']
        else:
            self.X_symmetry = np.zeros(len(X), dtype=bool)

        if self.lags is not None:
            #create (time-lagged) training data
            X_train, y_train = self.feat_eng.lag_training_data(X, target, lags=self.lags,
                                                               X_symmetry=self.X_symmetry)
        else:
            #do not time lag data
            n_samples = X[0].shape[0]
            X_train = np.array(X).reshape([n_samples, -1])
            y_train = target

        #set the training data, training size and bathc size for the online backprop step
        if not self.neural_net.batch_size == batch_size:
            self.neural_net.set_batch_size(batch_size)
        self.neural_net.n_train = X_train.shape[0]
        #standardize training data
        self.neural_net.X = (X_train - self.feat_mean) / self.feat_std
        self.neural_net.y = (y_train - self.output_mean) / self.output_std

        # train network for n_iter mini batches
        self.neural_net.train(n_iter, store_loss=True, sequential=sequential, verbose=verbose)

    def predict(self, X):
        """
        Make a stochastic prediction of the output y conditional on the
        input features X

        Parameters
        ----------
        X: the state at the current (time) step. If the ANN is conditioned on
        more than 1 (time-lagged) variable, X must be a list containing all
        variables at the current time step.

        Returns
        -------
        Prediction of the output y

        """
        if self.lags is not None:
            if not isinstance(X, list):
                X = [X]
            # append the current state X to the feature history
            self.feat_eng.append_feat(X)
            # get the feature history defined by the specified number of time lags.
            # Here, feat is an array with the same size as the neural network input layer
            feat = self.feat_eng.get_feat_history()
        else:
            feat = X

        # features were standardized during training, do so here as well
        feat = (feat - self.feat_mean) / self.feat_std
        # feed forward prediction step
        y = self.neural_net.feed_forward(feat.reshape([1, self.neural_net.n_in])).flatten()
        # transform y back to physical domain
        y = y * self.output_std + self.output_mean

        return y

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
