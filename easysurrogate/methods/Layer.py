"""
Class for a neural network Layer.
"""

import sys
import numpy as np
from scipy.stats import norm, bernoulli

from .batch_normalization import Batch_Normalization
from .Concatenate import Concatenate


class Layer:
    """
    Class for a neural network Layer.

    Method and notation convention:
        Aggarwal, Charu C. "Neural networks and deep learning."
        Springer 10 (2018): 978-3.
    """

    def __init__(self, n_neurons, activation, loss=None, bias=False,
                 batch_size=1, batch_norm=False, lamb=0.0, on_gpu=False,
                 n_softmax=0, **kwargs):
        """
        Create a Layer object.

        Parameters
        ----------
        n_neurons : int, optional
            The number of neurons per hidden layer. The default is 16.
        activation : string, optional
            The name of the activation function of the hidden layers.
            The default is 'tanh'.
        loss : string
            The name of the loss function.
        bias : boolean, optional
            Use a bias neuron. The default is False.
        batch_size : int, optional
            The size of the mini batch. The default is 1.
        batch_norm : boolean, optional
            Use batch normalization. The default is False.
        lamb : float, optional
            L2 weight regularization parameter. The default is 0.0.
        on_gpu : boolean, optional
            Train the neural network on a GPU using cupy. NOT IMPLEMENTED IN THIS VERSION.
            The default is False.
        n_softmax : int, optional
            The number of softmax layers attached to the output. The default is 0.

        Returns
        -------
        None.

        """

        self.n_neurons = n_neurons
        # self.r = r
        # self.n_layers = n_layers
        self.output_layer = False
        self.activation = activation
        self.loss = loss
        self.bias = bias
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.lamb = lamb
        self.n_softmax = n_softmax
        self.layer_rm1 = self.layer_rp1 = None
        self.trainable = True

        # #use either numpy or cupy via xp based on the on_gpu flag
        # global xp
        # if on_gpu == False:
        #     import numpy as xp
        # else:
        #     import cupy as xp

        self.on_gpu = False

        if self.bias:
            self.n_bias = 1
        else:
            self.n_bias = 0

        # the number of classes in each softmax layer
        if self.n_softmax > 0:
            self.n_bins = int(self.n_neurons / self.n_softmax)

        self.a = np.zeros([n_neurons, batch_size])
        self.h = np.zeros([n_neurons + self.n_bias, batch_size])
        self.delta_ho = np.zeros([n_neurons, batch_size])
        self.grad_Phi = np.zeros([n_neurons, batch_size])

        # if a kernel mixture network is used and this is the last layer:
        # store kernel means and standard deviations
        if loss == 'kernel_mixture':  # and r == n_layers:
            self.kernel_means = kwargs['kernel_means']
            self.kernel_stds = kwargs['kernel_stds']

        # if parametric relu activation is used, set the value of a (x if x>0; a*x otherwise)
        if activation == 'parametric_relu':
            self.relu_a = kwargs['relu_a']

        if batch_norm:
            self.bn = Batch_Normalization(self)

    def __call__(self, layer_rm1):
        self.layer_rm1 = layer_rm1
        layer_rm1.layer_rp1 = self

    def meet_the_neighbors(self, layer_rm1, layer_rp1):
        """
        Connect this layer to its neighbors

        Parameters
        ----------
        layer_rm1 : Layer object or None
            The layer before at index r - 1.
        layer_rp1 : Layer object or None
            The layer after at index r + 1.

        Returns
        -------
        None.

        """

        self.layer_rm1 = layer_rm1
        self.layer_rp1 = layer_rp1

        # set the r index (0 for inout, n_layers for output layer)
        if self.layer_rm1 is None:
            self.r = 0  # input layer: r = 0
        else:
            self.r = layer_rm1.r + 1  # r of previous layer + 1

        # if this is the output layer: n_layers = r
        if layer_rp1 is None:
            self.output_layer = True

        # fill the layer with neurons if it is not an input layer
        if self.layer_rm1 is not None:
            self.init_weights()

    def init_weights(self):
        """
        Initialize the weights and other related matrices of this layer

        Returns
        -------
        None.

        """
        # weights
        self.W = np.random.randn(self.layer_rm1.n_neurons + self.layer_rm1.n_bias,
                                 self.n_neurons) * np.sqrt(1.0 / self.layer_rm1.n_neurons)
        # loss gradient
        self.L_grad_W = np.zeros([self.layer_rm1.n_neurons + self.layer_rm1.n_bias, self.n_neurons])
        # momentum
        self.V = np.zeros([self.layer_rm1.n_neurons + self.layer_rm1.n_bias, self.n_neurons])
        # squared gradient
        self.A = np.zeros([self.layer_rm1.n_neurons + self.layer_rm1.n_bias, self.n_neurons])
        # L2 regularization
        self.Lamb = np.ones([self.layer_rm1.n_neurons + self.layer_rm1.n_bias,
                             self.n_neurons]) * self.lamb

        # do not apply regularization to the bias terms
        if self.bias:
            self.Lamb[-1, :] = 0.0

    def compute_output(self, batch_size, dropout=False, **kwargs):
        """
        Compute the output of the current layer in one shot using matrix -
        vector/matrix multiplication.

        Parameters
        ----------
        batch_size : int
            The batch size.

        Returns
        -------
        None.

        """

        a = np.dot(self.W.T, self.layer_rm1.h)
        if self.batch_norm:
            # overwrite a with normalized value
            a = self.bn.normalize(a)

        # apply activation to a
        if self.activation == 'linear':
            self.h = a
        elif self.activation == 'sigmoid':
            self.h = 1.0 / (1.0 + np.exp(-a))
        elif self.activation == 'relu':
            self.h = np.maximum(np.zeros([a.shape[0], a.shape[1]]), a)
        elif self.activation == 'leaky_relu':
            aa = np.copy(a)
            idx_lt0 = np.where(a <= 0.0)
            aa[idx_lt0[0], idx_lt0[1]] *= 0.01
            self.h = aa
        elif self.activation == 'parametric_relu':
            aa = np.copy(a)
            idx_lt0 = np.where(a <= 0.0)
            aa[idx_lt0[0], idx_lt0[1]] *= self.relu_a
            self.h = aa
        elif self.activation == 'softplus':
            self.h = np.log(1.0 + np.exp(a))
        elif self.activation == 'tanh':
            self.h = np.tanh(a)
        elif self.activation == 'hard_tanh':

            aa = np.copy(a)
            idx_gt1 = np.where(a >= 1.0)
            idx_ltm1 = np.where(a <= -1.0)
            aa[idx_gt1[0], idx_gt1[1]] = 1.0
            aa[idx_ltm1[0], idx_ltm1[1]] = -1.0

            self.h = aa

        else:
            print('Unknown activation type')
            sys.exit()

        if dropout:
            r = bernoulli.rvs(kwargs['dropout_prob'], size=self.h.shape)
            self.h *= r

        # add bias neuron output
        if self.bias:
            self.h = np.vstack([self.h, np.ones(batch_size)])
        self.a = a

        # compute the gradient of the activation function,
        self.compute_grad_Phi()

    def compute_grad_Phi(self):
        """
        Compute the gradient in the activation function Phi wrt its input

        Returns
        -------
        None.

        """

        if self.activation == 'linear':
            self.grad_Phi = np.ones([self.n_neurons, self.batch_size])
        elif self.activation == 'sigmoid':
            self.grad_Phi = self.h[0:self.n_neurons] * (1.0 - self.h[0:self.n_neurons])
        elif self.activation == 'relu':
            idx_lt0 = np.where(self.a < 0.0)
            self.grad_Phi = np.ones([self.n_neurons, self.batch_size])
            self.grad_Phi[idx_lt0[0], idx_lt0[1]] = 0.0
        elif self.activation == 'leaky_relu':
            idx_lt0 = np.where(self.a < 0.0)
            self.grad_Phi = np.ones([self.n_neurons, self.batch_size])
            self.grad_Phi[idx_lt0[0], idx_lt0[1]] = 0.01
        elif self.activation == 'parametric_relu':
            idx_lt0 = np.where(self.a < 0.0)
            self.grad_Phi = np.ones([self.n_neurons, self.batch_size])
            self.grad_Phi[idx_lt0[0], idx_lt0[1]] = self.relu_a
        elif self.activation == 'softplus':
            self.grad_Phi = 1.0 / (1.0 + np.exp(-self.a))
        elif self.activation == 'tanh':
            self.grad_Phi = 1.0 - self.h[0:self.n_neurons]**2
        elif self.activation == 'hard_tanh':
            idx = np.where(np.logical_and(self.a > -1.0, self.a < 1.0))
            self.grad_Phi = np.zeros([self.n_neurons, self.batch_size])
            self.grad_Phi[idx[0], idx[1]] = 1.0

    def compute_loss(self, h, y_i):
        """
        Compute the value of the loss function.


        Parameters
        ----------
        h : array
            The activation of the output layer.
        y_i : array
            The target data.

        Returns
        -------
        None.

        """

        # only compute if in an output layer
        if self.layer_rp1 is None:
            if self.loss == 'perceptron_crit':
                self.L_i = np.max([-y_i * h, 0.0])
            elif self.loss == 'hinge':
                self.L_i = np.max([1.0 - y_i * h, 0.0])
            elif self.loss == 'logistic':
                self.L_i = np.log(1.0 + np.exp(-y_i * h))
            elif self.loss == 'squared':
                self.L_i = (y_i - h)**2
            elif self.loss == 'cross_entropy':
                # compute values of the softmax layer
                # more than 1 (independent) softmax layer can be placed at the output
                o_i = []
                [o_i.append(np.exp(h_i) / np.sum(np.exp(h_i), axis=0))
                 for h_i in np.split(h, self.n_softmax)]
                self.o_i = np.concatenate(o_i)
                # cross entropy loss with a softmax layer
                try:
                    self.L_i = -np.sum(y_i * np.log(self.o_i))
                except RuntimeWarning:
                    # if a neuron does not fire, e.g. in the case of RelU activation,
                    # we get a RuntimeWarning. Just add a small constant in this case.
                    self.L_i = -np.sum(y_i * np.log(self.o_i + 1e-20))
            elif self.loss == 'binary_cross_entropy':
                self.L_i = - y_i * np.log(h) - (1 - y_i) * np.log(1 - h)
            elif self.loss == 'kernel_mixture' and self.n_softmax > 0:

                if y_i.ndim == 1:
                    y_i = y_i.reshape([1, y_i.size])

                self.L_i = 0.0
                idx = 0
                self.o_i = []
                self.p_i = []
                for h_i in np.split(h, self.n_softmax):
                    o_i = np.exp(h_i) / np.sum(np.exp(h_i), axis=0)
                    K_i = norm.pdf(y_i[idx], self.kernel_means[idx], self.kernel_stds[idx])
                    p_i = K_i * np.exp(h_i) / np.sum(K_i * np.exp(h_i), axis=0)
                    self.L_i = self.L_i - np.log(np.sum(o_i * K_i, axis=0))

                    self.o_i.append(o_i)
                    self.p_i.append(p_i)

                    idx += 1

                self.o_i = np.concatenate(self.o_i)
                self.p_i = np.concatenate(self.p_i)
            # user defined loss function
            elif hasattr(self.loss, '__call__'):
                # NOTE: in this case, not only the loss value, but also the
                # loss gradient of the last layer (delta_ho) is compute here
                # The user defined loss function expects arguments h, y_i,
                # h the prediction and y_i the data.
                self.L_i, self.delta_ho = self.loss(h, y_i)
            else:
                print('Cannot compute loss: unknown loss and/or activation function')
                sys.exit()

    def compute_delta_hy(self, norm=True):
        """
        Compute the gradient of the network output wrt the activation functions
        of this layer.

        norm : Boolean, optional, default is True.
               Compute the gradient of ||y||_2. If False it computes the gradient of
               y, if y is a scalar. If False and y is a vector, the resulting gradient is the
               column sum of the full Jacobian matrix.

        Returns
        -------
        None.

        """

        # if this layer is the output layer
        if self.layer_rp1 is None:
            # Using this computes the derivatives of column sums of the jacobian
            if not norm:
                self.delta_hy = np.ones([self.n_neurons, self.batch_size])
            else:
                # Using this computes the derivatives of the L2^2 norm of y
                # self.delta_hy = 2 * self.h
                # Using this computes the derivatives of the L2^2 norm of y
                if self.loss == 'cross_entropy':
                    o_i = []
                    [o_i.append(np.exp(h_i) / np.sum(np.exp(h_i), axis=0))
                     for h_i in np.split(self.h, self.n_softmax)]
                    o_i = np.concatenate(o_i)
                    self.delta_hy = o_i / np.linalg.norm(o_i)
                else:
                    self.delta_hy = self.h / np.linalg.norm(self.h)
        else:
            # get the delta_ho values of the next layer (layer r+1)
            delta_hy_rp1 = self.layer_rp1.delta_hy

            # get the grad_Phi values of the next layer
            grad_Phi_rp1 = self.layer_rp1.grad_Phi

            # the weight matrix of the next layer
            W_rp1 = self.layer_rp1.W

            self.delta_hy = np.dot(W_rp1, delta_hy_rp1 * grad_Phi_rp1)[0:self.n_neurons, :]

    def compute_delta_oo(self, y_i):
        """
        Initialize the value of delta_ho at the output layer. This is the gradient
        of the loss function wrt the output of the neural network.

        Parameters
        ----------
        y_i : array
            The target data.

        Returns
        -------
        None.

        """

        # if the neuron is in the output layer, initialze delta_oo
        if self.layer_rp1 is None:

            # compute the loss function
            self.compute_loss(self.h, y_i)

            h = self.h

            # for binary classification
            if self.loss == 'logistic' and self.activation == 'linear':

                self.delta_ho = -y_i * np.exp(-y_i * h) / (1.0 + np.exp(-y_i * h))

            elif self.loss == 'squared':  # and self.activation == 'linear':

                self.delta_ho = -2.0 * (y_i - h)
                # grad_loss = elementwise_grad(self.test)
                # self.delta_ho = grad_loss(self.h, y_i)

            # for multinomial classification
            elif self.loss == 'cross_entropy':
                # one-hot encoded data (y_i contains only 0's and 1's)
                if np.array_equal(y_i, y_i.astype(bool)):
                    # (see eq. 3.22 of Aggarwal book)
                    self.delta_ho = self.o_i - y_i
                # y_i is a more general probability mass function
                # delta_ho_i = sum_j(y_j * o_i) - y_i
                else:
                    self.delta_ho = np.array([(self.o_i[i] * y_i).sum(axis=0)
                                              for i in range(y_i.shape[0])])
                    self.delta_ho -= y_i
            # binary cross entropy loss
            elif self.loss == 'binary_cross_entropy':

                self.delta_ho = -1 / (h - (1 - y_i))

            # loss function for kernel mixture method
            elif self.loss == 'kernel_mixture' and self.n_softmax > 0:

                self.delta_ho = self.o_i - self.p_i

        else:
            # if the output layer is connected to another layer AFTER
            # it's own index r (at r + 1), take the gradient from that
            # layer. Allows to back propagate loss from one network to
            # the next as in GANs.
            self.delta_ho = self.layer_rp1.delta_ho

    def compute_delta_ho(self):
        """
        Compute the gradient of the loss function wrt the activation functions
        of this layer.

        Returns
        -------
        None.

        """
        # get the delta_ho values of the next layer (layer r+1)
        delta_ho_rp1 = self.layer_rp1.delta_ho

        # get the grad_Phi values of the next layer
        grad_Phi_rp1 = self.layer_rp1.grad_Phi

        # the weight matrix of the next layer
        W_rp1 = self.get_weights_next_layer()

        self.delta_ho = np.dot(W_rp1, delta_ho_rp1 * grad_Phi_rp1)[0:self.n_neurons, :]

    def get_weights_next_layer(self):

        if isinstance(self.layer_rp1, Concatenate):
            return self.layer_rp1.get_weights(self.r)
        else:
            return self.layer_rp1.W

    def compute_y_grad_W(self):
        """
        Compute the gradient of the network output wrt the weights of this layer.

        Returns
        -------
        None.

        """
        h_rm1 = self.layer_rm1.h
        delta_hy_grad_Phi = self.delta_hy * self.grad_Phi
        self.y_grad_W = np.dot(h_rm1, delta_hy_grad_Phi.T) / self.batch_size

    def compute_L_grad_W(self):
        """
        Compute the gradient of the loss function wrt the weights of this layer.

        Returns
        -------
        None.

        """

        if not self.batch_norm:
            h_rm1 = self.layer_rm1.h
            delta_ho_grad_Phi = self.delta_ho * self.grad_Phi
            self.L_grad_W = np.dot(h_rm1, delta_ho_grad_Phi.T) / self.batch_size
        else:
            self.L_grad_W = self.bn.compute_L_grad_W()

    def back_prop(self, y_i):
        """
        Perform the backpropogation operations of the current layer.

        Parameters
        ----------
        y_i : array
            The target data.

        Returns
        -------
        None.

        """

        # if self.r == self.n_layers:
        if self.output_layer:
            self.compute_delta_oo(y_i)
        else:
            self.compute_delta_ho()
        self.compute_L_grad_W()
