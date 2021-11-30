"""
==============================================================================
CLASS FOR A DEEP ACTIVE SUBSPACE SURROGATE
------------------------------------------------------------------------------
Author:
        W. Edeling
Method:
        Tripathy, Rohit, and Ilias Bilionis. "Deep active subspaces: A scalable
        method for high-dimensional uncertainty propagation."
        ASME 2019 International Design Engineering Technical Conferences and
        Computers and Information in Engineering Conference.
        American Society of Mechanical Engineers Digital Collection, 2019.
==============================================================================
"""

import numpy as np
import easysurrogate as es
from ..campaign import Campaign


class DAS_Surrogate(Campaign):
    """
    ANN_surrogate class for a standard feed forward neural network.
    """

    def __init__(self, **kwargs):
        print('Creating DAS_Surrogate Object')
        self.name = 'DAS Surrogate'
        # create a Feature_Engineering object
        self.feat_eng = es.methods.Feature_Engineering()

    ############################
    # START COMMON SUBROUTINES #
    ############################

    def train(self, feats, target, d, n_iter, test_frac=0.0,
              n_layers=2, n_neurons=100,
              activation='tanh',
              batch_size=64, lamb=0.0,
              standardize_X=True, standardize_y=True, **kwargs):
        """
        Perform backpropagation to train the DAS network

        Parameters
        ----------
        feats : array [n_samples, n_feats]
            The feature array.
        target : array
            The target data.
        d : integer
            The dimension of the active subspace.
        n_iter : integer
            The number of mini batch iterations.
        test_frac : float, optional
            Fraction of the data to use for testing. The default is 0.0.
        n_layers : integer, optional
            The number of hidden layers + the output layer. The default is 2.
        n_neurons : integer, optional
            The number of neurons per hidden layer. The default is 100.
        activation : string, optional
            The activation function to use. The default is 'tanh'. Other options include 'relu',
            'hard_tanh', 'leaky_relu', 'sigmoid', 'softplus'.
        batch_size : integer, optional
            The minibatch size. The default is 64.
        lamb : float, optional
            L2 weight regularization parameter. The default is 0.0.
        standardize_X : Boolean, optional
            Standardize the features. The default is True.
        standardize_y : Boolean, optional
            Standardize the output. The default is True.

        Returns
        -------
        None.

        """

        # test fraction
        self.test_frac = test_frac

        # The dimenions of the active subspace
        self.d = d

        # prepare the training data
        X_train, y_train, _, _ = self.feat_eng.get_training_data(
            feats, target, test_frac=test_frac, train_first=True)

        n_out = y_train.shape[1]

        # create the feed-forward ANN
        self.neural_net = es.methods.DAS_network(X_train, y_train, d,
                                                 n_layers=n_layers, n_neurons=n_neurons,
                                                 n_out=n_out,
                                                 loss='squared',
                                                 activation=activation, batch_size=batch_size,
                                                 lamb=lamb, decay_step=10**4, decay_rate=0.9,
                                                 standardize_X=standardize_X,
                                                 standardize_y=standardize_y,
                                                 save=False, **kwargs)

        print('===============================')
        print('Training Deep Active Subspace Neural Network...')

        # train network for n_iter mini batches
        self.neural_net.train(n_iter, store_loss=True)
        self.set_data_stats()

    def derivative(self, x, norm=True):
        """
        Compute a derivative of the network output f(x) with respect to the inputs x.

        Parameters
        ----------
        x : array
            A single feature vector of shape (n_in,) or (n_in, 1), where n_in is the
            number of input neurons.
        norm : Boolean, optional, default is True
            Compute the gradient of ||f||_2. If False it computes the gradient of
            f, if f is a scalar. If False and f is a vector, the resulting gradient is of the
            column sum of the full Jacobian matrix.

        Returns
        -------
        df_dx : array
            The derivatives [d||f||_2/dx_1, ..., d||f||_2/dx_n_in]

        """
        # check that x is of shape (n_in, ) or (n_in, 1)
        assert x.shape[0] == self.neural_net.n_in, \
        "x must be of shape (n_in,): %d != %d" % (x.shape[0], self.neural_net.n_in)

        if x.ndim > 1:
            assert x.shape[1] == 1, "Only pass 1 feature vector at a time"

        # set the batch size to 1 if not done already
        if not self.neural_net.batch_size == 1:
            self.neural_net.set_batch_size(1)

        # standardize the input (if inputs were not standardized, feat_mean=0 and feat_std=1)
        x = (x - self.feat_mean) / self.feat_std

        # feed forward and compute and the derivatives
        df_dx = self.neural_net.d_norm_y_dX(x.reshape([1, -1]), feed_forward=True, norm=norm)

        return df_dx

    def predict(self, feat):
        """
        Make a prediction with the Deep Active Subspace network at input parameter settings
        given by feat.

        Parameters
        ----------
        feat : array
               The feature array; the values of the input parameters.

        Returns
        -------
        array
            The prediction of the DAS neural net.

        """

        # features were standardized during training, do so here as well
        feat = (feat - self.feat_mean) / self.feat_std
        # feed forward prediction step
        y = self.neural_net.feed_forward(feat.reshape([1, self.neural_net.n_in])).flatten()
        # transform y back to physical domain
        y = y * self.output_std + self.output_mean

        return y

    def save_state(self):
        """
        Save the state of the DAS surrogate to a pickle file
        """
        state = self.__dict__
        super().save_state(state=state, name=self.name)

    def load_state(self):
        """
        Load the state of the DAS surrogate from file
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

    def get_dimensions(self):
        """
        Get some useful dimensions of the DAS surrogate. Returns a dict with the number
        of training samples (n_train), the number of data samples (n_samples),
        the number of test samples (n_test), the total number of input parameters (D),
        the dimension of the active subspace (d), and the size of the ouput quantity of
        interest (n_out).

        Returns
        -------
        dims : dict
            The dimensions dictionary.

        """

        dims = {}
        dims['n_train'] = self.feat_eng.n_train
        dims['n_samples'] = self.feat_eng.n_samples
        dims['n_test'] = dims['n_samples'] - dims['n_train']
        dims['D'] = self.neural_net.n_in
        dims['d'] = self.d
        dims['n_out'] = self.neural_net.n_out

        return dims
