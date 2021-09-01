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
              batch_size=64, lamb=0.0, **kwargs):

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
                                                 standardize_X=True, standardize_y=True,
                                                 save=False, **kwargs)

        print('===============================')
        print('Training Deep Active Subspace Neural Network...')

        # train network for n_iter mini batches
        self.neural_net.train(n_iter, store_loss=True)
        self.set_data_stats()

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

    def sensitivity_measures(self, feats):
        """
        Compute global derivative-based sensitivity measures using the derivative of
        squared L2 norm of the output, computing usoing back propagation. Integration
        of the derivatives over sthe stochastic space is done via MC on the provided
        input features in feats.

        Parameters
        ----------
        feats : array
            An array of input parameter values.

        Returns
        -------
        idx : array
            Indices corresponding to input variables, ordered from most to least
            influential.

        """

        # standardize the features
        feats = (feats - self.feat_std) / self.feat_std
        N = feats.shape[0]

        # test using measure based on wights of active subspace, see also sensitivity
        # paper by Paul Constantine.
        # if self.d == 1:
        #     idx = np.fliplr(np.argsort(np.abs(self.neural_net.layers[1].W.T)))
        #     print('Parameters ordered from most to least important:')
        #     print(idx)
        #     return idx

        # Set the batch size to 1
        self.neural_net.set_batch_size(1)
        # initialize the derivatives
        self.neural_net.d_norm_y_dX(feats[0].reshape([1, -1]))
        # compute the squared gradient
        norm_y_grad_x2 = self.neural_net.layers[0].delta_hy**2
        # compute the mean gradient
        mean = norm_y_grad_x2 / N
        # loop over all samples
        for i in range(1, N):
            # compute the next (squared) gradient
            self.neural_net.d_norm_y_dX(feats[i].reshape([1, -1]))
            norm_y_grad_x2 = self.neural_net.layers[0].delta_hy**2
            mean += norm_y_grad_x2 / N
        # order parameters from most to least influential based on the mean
        # squared gradient
        idx = np.fliplr(np.argsort(np.abs(mean).T))
        print('Parameters ordered from most to least important:')
        print(idx)
        return idx, mean
