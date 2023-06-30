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
                 param_specific_learn_rate=True, loss='squared',
                 activation='tanh', activation_out='linear', activation_das='linear',
                 n_softmax=0, n_layers=2, n_neurons=16,
                 bias=True, batch_size=1, batch_norm=False,
                 save=True, name='DAS', on_gpu=False,
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
        batch_norm : boolean or list of booleans, optional
            Use batch normalization in hidden layers. Not used in the DAS layer.
            The default is False.
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

        # the activation of the DAS layer
        self.activation_das = activation_das

        # set all the common parameters via the parent ANN class,
        # but overwrite the init_network subsroutine in this class
        super().__init__(X, y, alpha=alpha, decay_rate=decay_rate,
                         decay_step=decay_step, beta1=beta1, beta2=beta2,
                         lamb=lamb, n_out=n_out, loss=loss, activation=activation,
                         activation_out=activation_out, n_softmax=n_softmax,
                         n_layers=n_layers, n_neurons=n_neurons, bias=bias,
                         batch_size=batch_size, batch_norm=batch_norm,
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

        # add the input layer
        self.layers.append(Layer(self.n_in, 'linear',
                                 bias=False, batch_size=self.batch_size,
                                 batch_norm=False,
                                 lamb=self.lamb, on_gpu=self.on_gpu))

        # dimension of the full input space
        self.D = self.n_in

        # by default, the 1st layer does not have a bias neuron. This way
        # the orthogonal vectors are only related to the D inputs, and not the
        # D+1 inputs + 1 bias neuron, which does not make physical sense
        self.bias[0] = False

        # add the deep active subspace layer
        self.layers.append(
            DAS_Layer(
                self.d, self.D,
                bias=self.bias[1],
                activation=self.activation_das,
                batch_size=self.batch_size))

        # adjust layer_activation of the ANN superclass
        self.layer_activation[1] = self.activation_das

        # add the hidden layers
        for r in range(2, n_hidden):
            self.layers.append(Layer(self.n_neurons, self.layer_activation[r],
                                     self.loss, self.bias[r], batch_size=self.batch_size,
                                     batch_norm=self.batch_norm[r],
                                     lamb=self.lamb, on_gpu=self.on_gpu))

        # add the output layer
        self.layers.append(
            Layer(
                self.n_out,
                self.activation_out,
                self.loss,
                batch_size=self.batch_size,
                batch_norm=False,
                lamb=self.lamb,
                n_softmax=self.n_softmax,
                on_gpu=self.on_gpu,
                **kwargs))
