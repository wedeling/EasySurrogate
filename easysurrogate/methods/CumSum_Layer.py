"""
Cumulative sum layer
"""
import numpy as np

from .Layer import Layer


class CumSum_Layer(Layer):

    def __init__(self, n_neurons, r, n_layers, loss, batch_size):
        """
        Initialize a CumSum layer object.

        Parameters
        ----------
        n_neurons : int
            The number of neurons. Must match the number of neurons of the output layer..
        r : int
            The layer index.
        n_layers : int
            The number of layers in the network.
        loss : string
            The loss function.
        batch_size : int
            The batch size.

        Returns
        -------
        None.

        """
        super().__init__(n_neurons, r, n_layers, 'linear', loss,
                         bias=False, batch_size=batch_size)

    def compute_output(self, batch_size):
        self.h = np.cumsum(self.layer_rm1.h, axis=0)
        # compute the gradient of the activation function,
        self.compute_grad_Phi()

    def init_weights(self):
        """
        Initialize the weights and other related matrices of this layer. As this
        layer will be attached to the output, we assume the previous layers does
        not have bias neurons.

        Returns
        -------
        None.

        """
        # weights corresponding to a cumulative sum.
        self.W = np.triu(np.ones([self.layer_rm1.n_neurons, self.n_neurons]))
        # loss gradient
        self.L_grad_W = np.zeros([self.layer_rm1.n_neurons, self.n_neurons])
        # momentum
        self.V = np.zeros([self.layer_rm1.n_neurons, self.n_neurons])
        # squared gradient
        self.A = np.zeros([self.layer_rm1.n_neurons, self.n_neurons])
        # L2 regularization
        self.Lamb = np.ones([self.layer_rm1.n_neurons, self.n_neurons]) * self.lamb

    def compute_L_grad_W(self):
        """
        Compute the gradient of the loss function wrt the weights of this layer.

        Returns
        -------
        None.

        """
        return self.L_grad_W
