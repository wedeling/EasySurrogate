"""
Class for a neural network Layer.
"""
import sys
import numpy as np
from scipy.stats import norm


class Layer:
    """
    Class for a neural network Layer.

    Method and notation convention:
        Aggarwal, Charu C. "Neural networks and deep learning."
        Springer 10 (2018): 978-3.
    """

    def __init__(self, n_neurons, r, n_layers, activation, loss, bias=False,
                 batch_size=1, lamb=0.0, on_gpu=False,
                 n_softmax=0, **kwargs):
        """
        Create a Layer object.

        Parameters
        ----------
        n_neurons : int, optional
            The number of neurons per hidden layer. The default is 16.
        r : int
            The layer index, 0 being the input layer and n_layers the output layer.
        n_layers : int, optional
            The number of layers, not counting the input layer. The default is 2.
        activation : string, optional
            The name of the activation function of the hidden layers.
            The default is 'tanh'.
        loss : string, optional
            The name of the loss function. The default is 'squared'.
        bias : boolean, optional
            Use a bias neuron. The default is True.
        batch_size : int, optional
            The size of the mini batch. The default is 1.
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
        self.r = r
        self.n_layers = n_layers
        self.activation = activation
        self.loss = loss
        self.bias = bias
        self.batch_size = batch_size
        self.lamb = lamb
        self.n_softmax = n_softmax

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
        if loss == 'kernel_mixture' and r == n_layers:
            self.kernel_means = kwargs['kernel_means']
            self.kernel_stds = kwargs['kernel_stds']

        # if parametric relu activation is used, set the value of a (x if x>0; a*x otherwise)
        if activation == 'parametric_relu':
            self.relu_a = kwargs['relu_a']

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
        # if this layer is an input layer
        if self.r == 0:
            self.layer_rm1 = None
            self.layer_rp1 = layer_rp1
        # if this layer is an output layer
        elif self.r == self.n_layers:
            self.layer_rm1 = layer_rm1
            self.layer_rp1 = None
        # if this layer is hidden
        else:
            self.layer_rm1 = layer_rm1
            self.layer_rp1 = layer_rp1

        # fill the layer with neurons
        if self.r != 0:
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

    def compute_output(self, batch_size):
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

        # apply activation to a
        if self.activation == 'linear':
            self.h = a
        elif self.activation == 'sigmoid':
            self.h = 1.0 / (1.0 - np.exp(-a))
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
#                    p_i = K_i*o_i/np.sum(K_i*o_i, axis=0)
                    self.L_i = self.L_i - np.log(np.sum(o_i * K_i, axis=0))

                    self.o_i.append(o_i)
                    self.p_i.append(p_i)

                    idx += 1

                self.o_i = np.concatenate(self.o_i)
                self.p_i = np.concatenate(self.p_i)

            else:
                print('Cannot compute loss: unknown loss and/or activation function')
                sys.exit()

    def compute_delta_hy(self):
        """
        Compute the gradient of the network output wrt the activation functions
        of this layer.

        Returns
        -------
        None.

        """

        # if this layer is the output layer
        if self.layer_rp1 is None:
            # Using this computes the derivatives of column sums of the jacobian
            # self.delta_hy = np.ones([self.n_neurons, self.batch_size])
            # Using this computes the derivatives of the L2^2 norm of y
            self.delta_hy = 2 * self.h
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

                # (see eq. 3.22 of Aggarwal book)
                self.delta_ho = self.o_i - y_i

            elif self.loss == 'kernel_mixture' and self.n_softmax > 0:

                self.delta_ho = self.o_i - self.p_i

        else:
            print('Can only initialize delta_oo in output layer')
            sys.exit()

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
        W_rp1 = self.layer_rp1.W

        self.delta_ho = np.dot(W_rp1, delta_ho_rp1 * grad_Phi_rp1)[0:self.n_neurons, :]

    def compute_y_grad_W(self):
        """
        Compute the gradient of the network output wrt the weights of this layer.

        Returns
        -------
        None.

        """
        h_rm1 = self.layer_rm1.h
        delta_hy_grad_Phi = self.delta_hy * self.grad_Phi
        self.y_grad_W = np.dot(h_rm1, delta_hy_grad_Phi.T)

    def compute_L_grad_W(self):
        """
        Compute the gradient of the loss function wrt the weights of this layer.

        Returns
        -------
        None.

        """
        h_rm1 = self.layer_rm1.h
        delta_ho_grad_Phi = self.delta_ho * self.grad_Phi
        self.L_grad_W = np.dot(h_rm1, delta_ho_grad_Phi.T)

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

        if self.r == self.n_layers:
            self.compute_delta_oo(y_i)
        else:
            self.compute_delta_ho()
        self.compute_L_grad_W()
