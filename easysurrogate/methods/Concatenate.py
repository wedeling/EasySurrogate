"""
Concatenation layer
"""

import numpy as np


class Concatenate:
    """
    Concatenation Layer. Takes a user-specified list of neural network layers
    and concatenates them, such that they are presentented as a single
    layer to the next layer in the network. This layer has no trainable weights
    of its own. It copies the activations from the concatenated previous layers
    in a feed forward operation, and it copies the loss gradient from the next
    layer in a back propagation step.
    """

    def __init__(self):
        """
        Create a Concatenate object.

        Returns
        -------
        None.

        """
        # this layer has no bias neuron
        self.n_bias = 0
        self.bias = False
        # initialize the number of neurons to zero. Will be recomputed
        # when concatenation layers are specified.
        self.n_neurons = 0
        # Concatenate has no activation function
        self.activation = None
        self.name = 'Concatenation Layer'
        # Concatenate has no trainable weights & no batch normalization
        self.trainable = False
        self.batch_norm = False
        self.layer_rm1 = self.layer_rp1 = None

    def __call__(self, layers):
        """
        Specify the layers that are to be concatenated.

        Parameters
        ----------
        layers : list
            List of neural network layers.

        Returns
        -------
        None.

        """

        assert isinstance(layers, list), "concatenation layers must be supplied in a list"

        # the previous layers to be concatenated
        self.layer_rm1 = layers

        # set the layer_rp1 attribute (the next layer) for all concatenated layers
        for layer in layers:
            layer.layer_rp1 = self

        # the number of (virtual) neurons is the sum of the neurons in the
        # cancatenated layers
        n_neurons = [layer.n_neurons + layer.n_bias for layer in self.layer_rm1]
        self.n_neurons = np.sum(n_neurons)

        # the cumulative sum of the number of neurons. Used to split the
        # weight layer of the next layer into the contributions for each
        # previous cancatenated layers (see back_prop subroutine)
        self.cumsum_neurons = np.cumsum(n_neurons)

    def compute_output(self, batch_size, **kwargs):
        """
        Compute the output of the Concatenate layer. This concatenates the
        output of each concatenated layer.

        Parameters
        ----------
        batch_size : int
            The size of the mini batch.

        Returns
        -------
        None.

        """

        # concatenate the output of the layers in layer_rm1
        h = [layer.h for layer in self.layer_rm1]
        self.h = np.concatenate(h)

    def init_weights(self):
        """
        Concatenate is not trainable. Initialize the weights to an empty array.

        Returns
        -------
        None.

        """

        self.W = np.empty([0])

    def get_weights(self, r):
        """
        For a given concatenated layer (with index r) get the rows of the
        weight matrix of the layer after Concatenate (index j) that is
        associated to that layer. This is used to compute which the part loss
        gradient at layer j will flow to layer r.

        Parameters
        ----------
        r : int
            The index of a concatenated layer.

        Returns
        -------
        array, shape (n_neurons_r, n_neurons_j)
            The rows of the weight matrix of layer j (the layer after Concatenate)
            that is connected to the layer with index r.

        """

        # if it does not exist yet, compute the index of all concatenated layers
        if not hasattr(self, 'rm1'):
            self.rm1 = np.array([layer.r for layer in self.layer_rm1])

        # the location of r in rm1
        idx = np.where(r == self.rm1)[0][0]

        # return the correct rows of the weight matrix of the next layer
        return self.W_rp1[idx]

    def back_prop(self, y_i):
        """
        Back propagate the loss through Concatenate. This simply copies the
        loss gradient from the layer after Concatenate. It also splits the
        weight matrix of the next layer along the rows (axis=0) according
        to the size of the concatenated layers. This ensures that the 
        correct parts of the loss gradient are passed back to the
        concatenated layers (see also get_weights).

        Parameters
        ----------
        y_i : array
            The target data.

        Returns
        -------
        None.

        """

        # dl/dh from the next layer
        self.delta_ho = self.layer_rp1.delta_ho
        # dPhi/da from the next layer
        self.grad_Phi = self.layer_rp1.grad_Phi
        # The weight matrix of the next layer. Split rows according to
        # the size of the concatenated layers. Used in get_weights.
        self.W_rp1 = np.split(self.layer_rp1.W, self.cumsum_neurons, axis=0)
