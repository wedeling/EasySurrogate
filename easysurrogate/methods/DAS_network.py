"""
Deep active subspace surrogate.
"""

import numpy as np

from .DAS_Layer import DAS_Layer
from .Layer import Layer
from .NN import ANN


class DAS_network(ANN):
    """
    Deep active subspace surrogate.

    Method:
        Tripathy, Rohit, and Ilias Bilionis. "Deep active subspaces: A scalable
        method for high-dimensional uncertainty propagation."
        ASME 2019 International Design Engineering Technical Conferences and
        Computers and Information in Engineering Conference.
        American Society of Mechanical Engineers Digital Collection, 2019.
    """

    def __init__(self, X, y, d, alpha=0.001, decay_rate=1.0, decay_step=10**5,
                 beta1=0.9, beta2=0.999, lamb=0.0, n_out=1,
                 param_specific_learn_rate=True, loss='squared', activation='tanh',
                 activation_out='linear', n_softmax=0, n_layers=2, n_neurons=16,
                 bias=True, batch_size=1, save=True,
                 name='DAS', on_gpu=False,
                 standardize_X=True, standardize_y=True, **kwargs):
        """
        Initialize the Deep Active Subspace surrogate object.

        Parameters
        ----------
        X : array
            The input features.
        y : array
            The target data.
        d : int
            The assumed dimension of the active subspace.
        alpha : float, optional
            The learning rate. The default is 0.001.
        decay_rate : float, optional
            Factor multiplying the decay rate every decay_step iterations.
            The default is 1.0.
        decay_step : int, optional
            The number of training iterations after which decay_rate is lowered.
            The default is 10**5.
        beta1 : float, optional
            Momentum parameter controlling the moving average of the loss gradient.
            Used for the parameter-specific learning rate. The default is 0.9.
        beta2 : float, optional
            Parameter controlling the moving average of the squared gradient.
            Used for the parameter-specific learning rate. The default is 0.999.
        lamb : float, optional
            L2 weight regularization parameter. The default is 0.0.
        n_out : int, optional
            The number of output neurons. The default is 1.
        param_specific_learn_rate : boolean, optional
            Use parameter-specific learing rate. The default is True.
        loss : string, optional
            The name of the loss function. The default is 'squared'.
        activation : string, optional
            The name of the activation function of the hidden layers.
            The default is 'tanh'.
        activation_out : string, optional
            The name of the activation function of the output layer.
            The default is 'linear'.
        n_softmax : int, optional
            The number of softmax layers attached to the output. The default is 0.
        n_layers : int, optional
            The number of layers, not counting the input layer. The default is 2.
        n_neurons : int, optional
            The number of neurons per hidden layer. The default is 16.
        bias : boolean, optional
            Use a bias neuron. The default is True.
        batch_size : int, optional
            The size of the mini batch. The default is 1.
        save : boolean, optional
            Save the neural network to a pickle file after training.
            The default is True.
        name : string, optional
            The name of the neural network. The default is 'DAS'.
        on_gpu : boolean, optional
            Train the neural network on a GPU using cupy. NOT IMPLEMENTED IN THIS VERSION.
            The default is False.
        standardize_X : boolean, optional
            Standardize the features. The default is True.
        standardize_y : boolean, optional
            Standardize the target data. The default is True.


        Returns
        -------
        None.

        """
        # the dimension of the active subspace
        self.d = d

        # set all the common parameters via the parent ANN class,
        # but overwrite the init_network subsroutine in this class
        super().__init__(X, y, alpha=alpha, decay_rate=decay_rate,
                         decay_step=decay_step, beta1=beta1, beta2=beta2,
                         lamb=lamb, n_out=n_out, loss=loss, activation=activation,
                         activation_out=activation_out, n_softmax=n_softmax,
                         n_layers=n_layers, n_neurons=n_neurons, bias=bias,
                         batch_size=batch_size,
                         param_specific_learn_rate=param_specific_learn_rate,
                         save=save, on_gpu=on_gpu, name=name,
                         standardize_X=standardize_X, standardize_y=standardize_y,
                         **kwargs)

    def init_network(self, **kwargs):
        """
        Set up the network structure by creating the Layer objects and
        connecting them together. The second layer is a DAS_Layer.
        This subroutine overwrites a subroutine from the parent ANN class.

        Returns
        -------
        None.

        """

        self.layers = []

        n_hidden = self.n_layers
        if 'cumsum' in kwargs and kwargs['cumsum']:
            self.n_layers += 1

        # add the input layer
        self.layers.append(Layer(self.n_in, 0, self.n_layers, 'linear',
                                 self.loss, False, batch_size=self.batch_size,
                                 lamb=self.lamb, on_gpu=self.on_gpu))

        # add the deep active subspace layer
        self.layers.append(DAS_Layer(self.d, self.n_layers, True,
                                     batch_size=self.batch_size))

        # add the hidden layers
        for r in range(2, n_hidden):
            self.layers.append(Layer(self.n_neurons, r, self.n_layers, self.activation,
                                     self.loss, self.bias, batch_size=self.batch_size,
                                     lamb=self.lamb, on_gpu=self.on_gpu))

        # add the output layer
        self.layers.append(
            Layer(
                self.n_out,
                r + 1,
                self.n_layers,
                self.activation_out,
                self.loss,
                batch_size=self.batch_size,
                lamb=self.lamb,
                n_softmax=self.n_softmax,
                on_gpu=self.on_gpu,
                **kwargs))

        super().connect_layers()
        super().print_network_info()

    def feed_forward(self, X_i, batch_size=1):
        """
        Run the deep active subspace network forward.

        Parameters
        ----------
        X_i : array
            The feauture array, needs to have shape [batch size, number of features].
        batch_size : int, optional
            The bath size. The default is 1.

        Returns
        -------
        array
            The prediction of the deep active subspace neural network.
        """

        # set the features at the output of in the input layer
        self.layers[0].h = X_i.T

        for i in range(1, self.n_layers + 1):
            # compute the output on the layer using matrix-maxtrix multiplication
            self.layers[i].compute_output(batch_size)

        return self.layers[-1].h

    def batch(self, X_i, y_i, alpha=0.001, beta1=0.9, beta2=0.999):
        """
        Update the weights of the neural network and the weights of
        the Gram-Schmidt vectors using a mini batch.

        Parameters
        ----------
        X_i : array
            The input features of the mini batch.
        y_i : array
            The target data of the mini batch.
        alpha : float, optional
            The learning rate. The default is 0.001.
        beta1 : float, optional
            Momentum parameter controlling the moving average of the loss gradient.
            Used for the parameter-specific learning rate. The default is 0.9.
        beta2 : float, optional
            Parameter controlling the moving average of the squared gradient.
            Used for the parameter-specific learning rate. The default is 0.999.

        Returns
        -------
        None.

        """

        self.feed_forward(X_i, self.batch_size)
        self.back_prop(y_i)

        for r in range(1, self.n_layers + 1):

            layer_r = self.layers[r]

            # Deep active subspace layer
            if r == 1:
                # momentum
                layer_r.V = beta1 * layer_r.V + (1.0 - beta1) * layer_r.L_grad_Q
                # moving average of squared gradient magnitude
                layer_r.A = beta2 * layer_r.A + (1.0 - beta2) * layer_r.L_grad_Q**2
            # standard layer
            else:
                # momentum
                layer_r.V = beta1 * layer_r.V + (1.0 - beta1) * layer_r.L_grad_W
                # moving average of squared gradient magnitude
                layer_r.A = beta2 * layer_r.A + (1.0 - beta2) * layer_r.L_grad_W**2

            # select learning rate
            if not self.param_specific_learn_rate:
                # same alpha for all weights
                alpha_i = alpha
            # param specific learning rate
            else:
                # RMSProp
                alpha_i = alpha / (np.sqrt(layer_r.A + 1e-8))

            # gradient descent update step with L2 regularization
            if self.lamb > 0.0:
                layer_r.W = (1.0 - layer_r.Lamb * alpha_i) * layer_r.W - alpha_i * layer_r.V
            # without regularization
            else:
                # Deep active subspace layer
                if r == 1:
                    # update the Q weights
                    layer_r.Q = layer_r.Q - alpha_i * layer_r.V
                    # compute the weights W(Q) via Gram Schmidt
                    layer_r.compute_weights()
                # standard layer
                else:
                    layer_r.W = layer_r.W - alpha_i * layer_r.V
