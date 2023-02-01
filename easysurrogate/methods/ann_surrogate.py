"""
==============================================================================
CLASS FOR A ARTIFICIAL NEURAL NETWORK
------------------------------------------------------------------------------
Author: W. Edeling
==============================================================================
"""

import easysurrogate as es
from ..campaign import Campaign


class ANN_Surrogate(Campaign):
    """
    ANN_surrogate class for a standard feed forward neural network.
    """

    def __init__(self, **kwargs):
        print('Creating ANN_Surrogate Object')
        self.name = 'ANN Surrogate'
        # create a Feature_Engineering object
        self.feat_eng = es.methods.Feature_Engineering()

    ############################
    # START COMMON SUBROUTINES #
    ############################

    def train(self, feats, target, n_iter, lags=None, local=False,
              test_frac=0.0,
              n_layers=2, n_neurons=100,
              loss='squared',
              activation='tanh',
              learning_rate = 0.001, decay_rate = 0.9, beta1 = 0.9,
              batch_size=64, lamb=0.0,
              standardize_X=True, standardize_y=True,
              dropout=False, **kwargs):
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
        loss : string, optional
            The name of the loss function. The default is 'squared'.
        activation : Type of activation function. The default is 'tanh'.
        learing_rate : the baseline learning rate. Is made parameter specific
                       via RMSProp. Th edefault is 0.001.
        decay_rate : float, optional
                     Factor multiplying the decay rate every decay_step 
                     iterations. Default is 0.9.
        beta1 : float, optional
                Momentum parameter controlling the moving average of the loss gradient.
                Used for the parameter-specific learning rate. The default is 0.9.
        batch_size : Mini batch size. The default is 64.
        lamb : L2 regularization parameter. The default is 0.0.
        dropout : Boolean flag for use of dropout regularization. 

        Returns
        -------
        None.

        """

        # time lags
        self.lags = lags

        # flag if the surrogate is to be applied locally or not
        self.local = local

        # test fraction
        self.test_frac = test_frac

        # loss function
        self.loss = loss

        # prepare the training data
        X_train, y_train, _, _ = self.feat_eng.get_training_data(
            feats, target, lags=lags, local=local, test_frac=test_frac, train_first=True)
        self.max_lag = self.feat_eng.max_lag

        # number of output neurons
        n_out = y_train.shape[1]

        # create the feed-forward ANN
        self.neural_net = es.methods.ANN(X=X_train, y=y_train,
                                         n_layers=n_layers, n_neurons=n_neurons,
                                         n_out=n_out,
                                         loss=loss,
                                         activation=activation, batch_size=batch_size,
                                         alpha=learning_rate,
                                         lamb=lamb, decay_step=10**4, 
                                         decay_rate=decay_rate, beta1=beta1,
                                         standardize_X=standardize_X,
                                         standardize_y=standardize_y,
                                         save=False,
                                         **kwargs)

        print('===============================')
        print('Training Artificial Neural Network...')

        # train network for n_iter mini batches
        self.neural_net.train(n_iter, store_loss=True, dropout=dropout,
                              **kwargs)
        self.set_data_stats()
        if lags is not None:
            self.feat_eng.initial_condition_feature_history(feats)

    def derivative(self, x, norm=True, layer_idx=0):
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
        layer_idx : int, optional, default is 0.
            Index for the layer of which to return the derivative. Default is 0, the input layer.

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
        if self.neural_net.batch_size != 1:
            self.neural_net.set_batch_size(1)

        # standardize the input (if inputs were not standardized, feat_mean=0 and feat_std=1)
        x = (x - self.feat_mean) / self.feat_std

        # feed forward and compute and the derivatives
        df_dx = self.neural_net.d_norm_y_dX(x.reshape([1, -1]), feed_forward=True, norm=norm,
                                            layer_idx=layer_idx)

        return df_dx

    def train_online(self, n_iter=1, batch_size=1, verbose=False, sequential=False):
        """
        Perform online training, i.e. backpropagation while the surrogate is coupled
        to the macroscopic governing equations.

        Source:
        Rasp, "Coupled online learning as a way to tackle instabilities and biases
        in neural network parameterizations: general algorithms and Lorenz 96
        case study", 2020.

        Parameters
        ----------
        feats : feature array, or list of different feature arrays
        target : the target data
        n_iter : number of back propagation iterations
        batch_size : Mini batch size. Default is 1.
        verbose: print loss to screen during back propagation. Default is False.
        sequential: do not randomly sample the training data, Default is False.

        Returns
        -------
        None.

        """

        X_train, y_train = self.feat_eng.get_online_training_data(n_in=self.neural_net.n_in,
                                                                  n_out=self.neural_net.n_out)

        # set the training data, training size and batch size for the online backprop step
        if self.neural_net.batch_size != batch_size:
            self.neural_net.set_batch_size(batch_size)
        self.neural_net.n_train = X_train.shape[0]
        # standardize training data
        self.neural_net.X = (X_train - self.feat_mean) / self.feat_std
        self.neural_net.y = (y_train - self.output_mean) / self.output_std

        # train network for n_iter mini batches
        self.neural_net.train(n_iter, store_loss=True, sequential=sequential, verbose=verbose)

    def generate_online_training_data(self, feats, LR_before, LR_after, HR_before, HR_after):
        """
        Compute the features and the target data for an online training step. Results are
        stored internally, and used within the 'train_online' subroutine.

        Source:
        Rasp, "Coupled online learning as a way to tackle instabilities and biases
        in neural network parameterizations: general algorithms and Lorenz 96
        case study", 2020.

        Parameters
        ----------
        feats : array or list of arrays
            The input features
        LR_before : array
            Low resolution state at previous time step.
        LR_after : array
            Low resolution state at current time step.
        HR_before : array
            High resolution state at previous time step.
        HR_after : array
            High resolution state at current time step.

        Returns
        -------
        None.

        """

        self.feat_eng.generate_online_training_data(feats, LR_before, LR_after, HR_before, HR_after)

    def set_online_training_parameters(self, tau_nudge, dt_LR, window_length):
        """
        Stores parameters required for online training.

        Parameters
        ----------
        tau_nudge : float
            Nudging time scale.
        dt_LR : float
            Time step low resolution model.
        window_length : int
            The length of the moving window in which online features are stored.

        Returns
        -------
        None.

        """
        self.feat_eng.set_online_training_parameters(tau_nudge, dt_LR, window_length)

    def predict(self, feat):
        """
        Make a prediction f(feat). Here, f is given by ANN_Surrogate_feed_foward.

        Parameters
        ----------
        feat : array of list of arrays
               The feature array of a list of feature arrays on which to evaluate the surrogate.

        Returns
        -------
        array
            the prediction of the neural net.

        """

        # feat_eng._predict handles the preparation of the features and returns
        # self._feed_forward(X)
        return self.feat_eng._predict(feat, self._feed_forward)

    def _feed_forward(self, feat):
        """
        A feed forward run of the ANN. This is the only part of prediction that is specific
        to ANN_Surrogate.

        Parameters
        ----------
        feat : array of list of arrays
               The feature array of a list of feature arrays on which to evaluate the surrogate.

        Returns
        -------
        y : array
            the prediction of the neural net.

        """

        # if features were standardized during training, do so here as well
        feat = (feat - self.feat_mean) / self.feat_std
        if self.loss == 'cross_entropy':
            # y = the probability mass function at the output layer
            y, _, _ = self.neural_net.get_softmax(feat.reshape([1, self.neural_net.n_in]))
        else:
            # feed forward prediction step
            y = self.neural_net.feed_forward(feat.reshape([1, self.neural_net.n_in])).flatten()
            # transform y back to physical domain
            y = y * self.output_std + self.output_mean

        return y

    def save_state(self):
        """
        Save the state of the ANN surrogate to a pickle file
        """
        state = self.__dict__
        super().save_state(state=state, name=self.name)

    def load_state(self):
        """
        Load the state of the ANN surrogate from file
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
        Get some useful dimensions of the ANN surrogate. Returns a dict with the number
        of training samples (n_train), the number of data samples (n_samples),
        the number of test samples (n_test), the number of input neurons (n_in),
        and the number of output_neurons (n_out).

        Returns
        -------
        dims : dict
            The dimensions dictionary.

        """

        dims = {}
        dims['n_train'] = self.feat_eng.n_train
        dims['n_samples'] = self.feat_eng.n_samples
        dims['n_test'] = dims['n_samples'] - dims['n_train']
        dims['n_in'] = self.neural_net.n_in
        dims['n_out'] = self.neural_net.n_out

        return dims
