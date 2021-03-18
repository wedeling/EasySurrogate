from .Neuron import Neuron
import numpy as np
from scipy.stats import norm


class Layer:

    def __init__(self, n_neurons, r, n_layers, activation, loss, bias=False,
                 neuron_based_compute=False, batch_size=1, lamb=0.0, on_gpu=False,
                 n_softmax=0, **kwargs):

        self.n_neurons = n_neurons
        self.r = r
        self.n_layers = n_layers
        self.activation = activation
        self.loss = loss
        self.bias = bias
        self.neuron_based_compute = neuron_based_compute
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

    # connect this layer to its neighbors
    def meet_the_neighbors(self, layer_rm1, layer_rp1):
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
            self.seed_neurons()

    # initialize the neurons of this layer
    def seed_neurons(self):

        # initialize the weight, gradient and momentum matrix
        #        self.W = np.zeros([self.layer_rm1.n_neurons + self.layer_rm1.n_bias, self.n_neurons])
        self.W = np.random.randn(self.layer_rm1.n_neurons + self.layer_rm1.n_bias,
                                 self.n_neurons) * np.sqrt(1.0 / self.layer_rm1.n_neurons)
        self.L_grad_W = np.zeros([self.layer_rm1.n_neurons + self.layer_rm1.n_bias, self.n_neurons])
        self.V = np.zeros([self.layer_rm1.n_neurons + self.layer_rm1.n_bias, self.n_neurons])
        self.A = np.zeros([self.layer_rm1.n_neurons + self.layer_rm1.n_bias, self.n_neurons])
        self.Lamb = np.ones([self.layer_rm1.n_neurons + self.layer_rm1.n_bias,
                             self.n_neurons]) * self.lamb

        # do not apply regularization to the bias terms
        if self.bias:
            self.Lamb[-1, :] = 0.0

        if self.neuron_based_compute:
            neurons = []

            for j in range(self.n_neurons):
                neurons.append(
                    Neuron(
                        self.activation,
                        self.loss,
                        self.layer_rm1,
                        self,
                        self.layer_rp1,
                        j))

            for j in range(self.n_neurons, self.n_neurons + self.n_bias):
                neurons.append(Neuron('bias', self.loss, self.layer_rm1, self, self.layer_rp1, j))

            self.neurons = neurons

    # return the output of the current layer, computed locally at each neuron
    def compute_output_local(self):
        for i in range(self.n_neurons + self.n_bias):
            self.neurons[i].compute_h()

        # compute the gradient of the activation function,
        self.compute_grad_Phi()

    # compute the output of the current layer in one shot using matrix -
    # vector/matrix multiplication
    def compute_output(self, batch_size):

        a = np.dot(self.W.T, self.layer_rm1.h)

        # apply activation to a
        if self.activation == 'linear':
            self.h = a
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
            import sys
            sys.exit()

        # add bias neuron output
        if self.bias:
            self.h = np.vstack([self.h, np.ones(batch_size)])
        self.a = a

        # compute the gradient of the activation function,
        self.compute_grad_Phi()

    # compute the gradient in the activation function Phi wrt its input
    def compute_grad_Phi(self):

        if self.activation == 'linear':
            self.grad_Phi = np.ones([self.n_neurons, self.batch_size])
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

    # compute the value of the loss function
    def compute_loss(self, h, y_i):

        # h = self.h

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
                import sys
                sys.exit()

    # compute the gradient of the output wrt the activation functions of this layer
    def compute_delta_hy(self):

        # if this layer is the output layer
        if self.layer_rp1 is None:
            self.delta_hy = np.ones([self.n_neurons, self.batch_size])
        else:
            # get the delta_ho values of the next layer (layer r+1)
            delta_hy_rp1 = self.layer_rp1.delta_hy

            # get the grad_Phi values of the next layer
            grad_Phi_rp1 = self.layer_rp1.grad_Phi

            # the weight matrix of the next layer
            W_rp1 = self.layer_rp1.W

            self.delta_hy = np.dot(W_rp1, delta_hy_rp1 * grad_Phi_rp1)[0:self.n_neurons, :]

    def test(self, h, y_i):
        return (y_i - h)**2

    # initialize the value of delta_ho at the output layer
    def compute_delta_oo(self, y_i):

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

            elif self.loss == 'kernel_mixture' and self.n_softmax == 0:

                self.delta_ho = -self.kernels / self.sum_kernels_w + 1.0 / self.sum_w

            elif self.loss == 'kernel_mixture' and self.n_softmax > 0:

                #                self.delta_ho = self.o_i*(1.0 - self.kernels/self.sum_kernels_Pr)
                self.delta_ho = self.o_i - self.p_i

            elif self.loss == 'custom':

                alpha = 0.95
                self.delta_ho = -2.0 * (1.0 - alpha) * (y_i - h) + 2.0 * alpha * (h + self.udv)

#                #Raissi's example
#                dt = 0.01;
#                self.delta_ho = -3.0*dt*(y_i - self.udh)

        else:
            print('Can only initialize delta_oo in output layer')
            import sys
            sys.exit()

    # compute the gradient of the loss function wrt the activation functions of this layer
    def compute_delta_ho(self):
        # get the delta_ho values of the next layer (layer r+1)
        delta_ho_rp1 = self.layer_rp1.delta_ho

        # get the grad_Phi values of the next layer
        grad_Phi_rp1 = self.layer_rp1.grad_Phi

        # the weight matrix of the next layer
        W_rp1 = self.layer_rp1.W

        self.delta_ho = np.dot(W_rp1, delta_ho_rp1 * grad_Phi_rp1)[0:self.n_neurons, :]

    # compute the gradient of the output wrt the weights of this layer
    def compute_y_grad_W(self):
        h_rm1 = self.layer_rm1.h

        delta_hy_grad_Phi = self.delta_hy * self.grad_Phi

        self.y_grad_W = np.dot(h_rm1, delta_hy_grad_Phi.T)

    # compute the gradient of the loss function wrt the weights of this layer
    def compute_L_grad_W(self):
        h_rm1 = self.layer_rm1.h

        delta_ho_grad_Phi = self.delta_ho * self.grad_Phi

        self.L_grad_W = np.dot(h_rm1, delta_ho_grad_Phi.T)

    # perform the backpropogation operations of the current layer
    def back_prop(self, y_i):

        if not self.neuron_based_compute:
            if self.r == self.n_layers:
                self.compute_delta_oo(y_i)
                self.compute_L_grad_W()
            else:
                self.compute_delta_ho()
                self.compute_L_grad_W()
        else:
            if self.r == self.n_layers:
                # initialize delta_oo
                for i in range(self.n_neurons):
                    self.neurons[i].compute_delta_oo(y_i)
                    self.neurons[i].compute_L_grad_W()
            else:
                for i in range(self.n_neurons):
                    self.neurons[i].compute_delta_ho()
                    self.neurons[i].compute_L_grad_W()

    # set a user defined value. To be used in the
    # output layer for custom loss functions and loss function gradients
    def set_user_defined_value(self, val):
        self.udv = val
