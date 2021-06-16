"""
Class for an artificial neural network.
"""

import pickle
import tkinter as tk
from tkinter import filedialog
import numpy as np
# from scipy.stats import rv_discrete
# import h5py

from .Layer import Layer


class ANN:
    """
    Class for an artificial neural network.
    """

    def __init__(self, X, y, alpha=0.001, decay_rate=1.0, decay_step=10**5, beta1=0.9,
                 beta2=0.999, lamb=0.0, n_out=1, loss='squared', activation='tanh',
                 activation_out='linear', n_softmax=0, n_layers=2, n_neurons=16,
                 bias=True, batch_size=1, param_specific_learn_rate=True,
                 save=True, on_gpu=False, name='ANN',
                 standardize_X=True, standardize_y=True, **kwargs):
        """
        Initialize the Artificial Neural Network object.

        Parameters
        ----------
        X : array
            The input features.
        y : array
            The target data.
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
        param_specific_learn_rate : boolean, optional
            Use parameter-specific learing rate. The default is True.
        save : boolean, optional
            Save the neural network to a pickle file after training.
            The default is True.
        on_gpu : boolean, optional
            Train the neural network on a GPU using cupy. NOT IMPLEMENTED IN THIS VERSION.
            The default is False.
        name : string, optional
            The name of the neural network. The default is 'ANN'.
        standardize_X : boolean, optional
            Standardize the features. The default is True.
        standardize_y : boolean, optional
            Standardize the target data. The default is True.

        Returns
        -------
        None.

        """

        # the features
        self.X = X

        # number of training data points
        self.n_train = X.shape[0]

        # the training outputs
        self.y = y

        # number of input nodes
        try:
            self.n_in = X.shape[1]
        except IndexError:
            self.n_in = 1

        # #use either numpy or cupy via xp based on the on_gpu flag
        # global xp
        # if on_gpu == False:
        #     import numpy as xp
        # else:
        #     import cupy as xp

        # self.on_gpu = on_gpu
        self.on_gpu = False

        # standardize the training data
        if standardize_X:

            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.X = (X - self.X_mean) / self.X_std

        if standardize_y:
            self.y_mean = np.mean(y, axis=0)
            self.y_std = np.std(y, axis=0)
            self.y = (y - self.y_mean) / self.y_std
        self.standardize_X = standardize_X
        self.standardize_y = standardize_y

        # number of layers (hidden + output)
        self.n_layers = n_layers

        # number of neurons in a hidden layer
        self.n_neurons = n_neurons

        # number of output neurons
        self.n_out = n_out

        # use bias neurons
        self.bias = bias

        # loss function type
        self.loss = loss

        # training rate
        self.alpha = alpha

        # L2 regularization parameter
        self.lamb = lamb

        # the rate of decay and decay step for alpha
        self.decay_rate = decay_rate
        self.decay_step = decay_step

        # momentum parameter
        self.beta1 = beta1

        # squared gradient parameter
        self.beta2 = beta2

        # use parameter specific learning rate
        self.param_specific_learn_rate = param_specific_learn_rate

        # activation function of the hidden layers
        self.activation = activation

        # activation function of the output layer
        self.activation_out = activation_out

        # number of sofmax layers
        self.n_softmax = n_softmax

        # save the neural network after training
        self.save = save
        self.name = name

        # additional variables/dicts etc that must be stored in the ann object
        self.aux_vars = kwargs

        # size of the mini batch used in stochastic gradient descent
        self.batch_size = batch_size

        self.loss_vals = []

        self.init_network(**kwargs)

    def init_network(self, **kwargs):
        """
        Set up the network structure by creating the Layer objects and
        connecting them together.

        Returns
        -------
        None.

        """

        self.layers = []

        # add the input layer
        self.layers.append(Layer(self.n_in, 0, self.n_layers, 'linear',
                                 self.loss, self.bias, batch_size=self.batch_size,
                                 lamb=self.lamb, on_gpu=self.on_gpu))

        # add the hidden layers
        for r in range(1, self.n_layers):
            self.layers.append(Layer(self.n_neurons, r, self.n_layers, self.activation,
                                     self.loss, self.bias, batch_size=self.batch_size,
                                     lamb=self.lamb, on_gpu=self.on_gpu))

        # add the output layer
        self.layers.append(
            Layer(
                self.n_out,
                self.n_layers,
                self.n_layers,
                self.activation_out,
                self.loss,
                batch_size=self.batch_size,
                lamb=self.lamb,
                n_softmax=self.n_softmax,
                on_gpu=self.on_gpu,
                **kwargs))

        self.connect_layers()

        self.print_network_info()

    def connect_layers(self):
        """
         Connect each layer in the neural network with its neighbours

        Returns
        -------
        None.

        """

        self.layers[0].meet_the_neighbors(None, self.layers[1])
        self.layers[-1].meet_the_neighbors(self.layers[-2], None)

        for i in range(1, self.n_layers):
            self.layers[i].meet_the_neighbors(self.layers[i - 1], self.layers[i + 1])

    def feed_forward(self, X_i, batch_size=1):
        """
        Run the network forward.

        Parameters
        ----------
        X_i : array
            The feauture array, needs to have shape [batch size, number of features].
        batch_size : int, optional
            The bath size. The default is 1.

        Returns
        -------
        array
            The prediction of the neural network.

        """

        # set the features at the output of in the input layer
        if not self.bias:
            self.layers[0].h = X_i.T
        else:
            self.layers[0].h = np.ones([self.n_in + 1, batch_size])
            self.layers[0].h[0:self.n_in, :] = X_i.T

        for i in range(1, self.n_layers + 1):
            # compute the output on the layer using matrix-maxtrix multiplication
            self.layers[i].compute_output(batch_size)

        return self.layers[-1].h

    def get_softmax(self, X_i):
        """
        Get the output of the softmax layer.

        Parameters
        ----------
        X_i : array
            The input features.

        Returns
        -------
        probs : array
            the probabilities of the softmax layer.
        idx_max : int
            The softmax output with the highest probability.

        """
        # feed forward features X_i
        h = self.feed_forward(X_i, batch_size=1)

        probs = []
        idx_max = []
        # rvs = []

        # split the output of the last layer over the number of softmax layers
        for h_i in np.split(h, self.n_softmax):
            # compute the softmax probabilities.
            o_i = np.exp(h_i) / np.sum(np.exp(h_i), axis=0)
            o_i = o_i / np.sum(o_i)
            probs.append(o_i)
            # the softmax output w
            idx_max.append(np.argmax(o_i))

            # draw a random sample from the discrete distribution o_i.
            # TODO: Slow for some reason. Find faster implementation.
            # pmf = rv_discrete(values=(np.arange(o_i.size), o_i.flatten()))
            # rvs.append(pmf.rvs())

        # return values and index of highest probability and random samples from pmf
        return probs, idx_max, None

    def d_norm_y_dX(self, X_i, batch_size=1, feed_forward=True):
        """
        Compute the derivatives of the squared L2 norm of the output wrt
        the inputs.

        Parameters
        ----------
        X_i : array
            The input features.
        batch_size : int, optional
            The batch size. The default is 1.

        Returns
        -------
        array
            The derivatives [d||y||^2_2/dX_1, ..., d||y||^2_2/dX_n_in]

        """
        if feed_forward:
            self.feed_forward(X_i, batch_size=batch_size)

        for i in range(self.n_layers, -1, -1):
            self.layers[i].compute_delta_hy()
            # also compute the gradient of the output wrt the weights
            if i > 0:
                self.layers[i].compute_y_grad_W()

        # delta_hy of the input layer = the derivative of the normed output
        return self.layers[0].delta_hy

    def back_prop(self, y_i):
        """
        Back-propagation algorithm to find gradient of the loss function with respect
        to the weights of the neural network.

        Parameters
        ----------
        y_i : array
            The target data on which to evaluate the loss funcion.

        Returns
        -------
        None.

        """

        # start back propagation over hidden layers, starting with output layer
        for i in range(self.n_layers, 0, -1):
            self.layers[i].back_prop(y_i)

    def batch(self, X_i, y_i, alpha=0.001, beta1=0.9, beta2=0.999):
        """
        Update the weights using a mini batch.

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

                # Adam
                #alpha_t = alpha*np.sqrt(1.0 - beta2**t)/(1.0 - beta1**t)
                #alpha_i = alpha_t/(np.sqrt(layer_r.A + 1e-8))

            # gradient descent update step with L2 regularization
            if self.lamb > 0.0:
                layer_r.W = (1.0 - layer_r.Lamb * alpha_i) * layer_r.W - alpha_i * layer_r.V
            # without regularization
            else:
                layer_r.W = layer_r.W - alpha_i * layer_r.V

                # Nesterov momentum
                # layer_r.W += -alpha*beta1*layer_r.V

    # train the neural network
    def train(
            self,
            n_batch,
            store_loss=False,
            sequential=False,
            verbose=True):
        """
        Train the neural network using stochastic gradient descent.

        Parameters
        ----------
        n_batch : int
            The number of mini-batch iterations.
        store_loss : boolean, optional
            Store the values of the loss function. The default is False.
        sequential : boolean, optional
            Sample a sequential slab of data, starting from a random point.
            The default is False.
        verbose : boolean, optional
            Print information to screen while training. The default is True.

        Returns
        -------
        None.

        """

        for i in range(n_batch):

            # select a random training instance (X, y)
            if not sequential:
                rand_idx = np.random.randint(0, self.n_train, self.batch_size)
            # select a random starting point, and use sequential data from there
            else:
                if self.n_train > self.batch_size:
                    start = np.random.randint(0, self.n_train - self.batch_size, 1)
                else:
                    start = 0
                rand_idx = np.arange(start, start + self.batch_size)

            # compute learning rate
            alpha = self.alpha * self.decay_rate**(np.int(i / self.decay_step))

            # run the batch
            self.batch(
                self.X[rand_idx],
                self.y[rand_idx].T,
                alpha=alpha,
                beta1=self.beta1,
                beta2=self.beta2)

            # store the loss value
            if store_loss:
                # l = 0.0
                # for k in range(self.n_out):
                # l += self.layers[-1].L_i

                l = self.layers[-1].L_i

                if np.mod(i, 1000) == 0:
                    loss_i = np.mean(l)
                    if verbose:
                        print('Batch', i, 'learning rate', alpha, 'loss:', loss_i)
                    # note: appending a cupy value to a list is inefficient -
                    # if done every iteration
                    # it will slow down executing significantly
                    self.loss_vals.append(loss_i)

        if self.save:
            self.save_ANN()

    def save_ANN(self, file_path="", store_data=False):
        """
        Save the neural network to a picke file.

        Parameters
        ----------
        file_path : string, optional
            The full path of the pickle file. The default is "", in which case a
            filedialog window opens.
        store_data : boolean, optional
            Store the training data as well. The default is False, since storing a
            large amount of data in a pickel file is inefficient.

        Returns
        -------
        None.

        """
        if len(file_path) == 0:

            root = tk.Tk()
            root.withdraw()

            file = filedialog.asksaveasfile(title="Save network",
                                            mode='wb', defaultextension=".pickle")
        else:
            file = open(file_path, 'wb')

        print('Saving ANN to', file.name)

        if store_data:
            # store everything, also data
            pickle.dump(self.__dict__, file)
        else:
            tmp = self.__dict__.copy()
            # do not store data to pickle
            tmp['X'] = []
            tmp['y'] = []
            pickle.dump(tmp, file)

        file.close()

    def load_ANN(self, file_path=""):
        """
        Load the neural network from a pickle file.

        Parameters
        ----------
        file_path : string, optional
            The full path of the pickle file. The default is "", in which case a
            filedialog window opens.

        Returns
        -------
        None.

        """

        # select file via GUI is file_path is not specified
        if len(file_path) == 0:

            root = tk.Tk()
            root.withdraw()

            file_path = filedialog.askopenfilename(title="Open neural network",
                                                   filetypes=(('pickle files', '*.pickle'),
                                                              ('All files', '*.*')))

        print('Loading ANN from', file_path)

        file = open(file_path, 'rb')
        self.__dict__ = pickle.load(file)
        file.close()

        if self.__dict__['X'] == []:
            print('===============================')
            print('**Warning: ANN was saved without training data**')
            print('===============================')

        self.print_network_info()

    def set_batch_size(self, batch_size):
        """
        Set the batch size in all the layers.

        Parameters
        ----------
        batch_size : int
            The batch size.

        Returns
        -------
        None.

        """

        self.batch_size = batch_size

        for i in range(self.n_layers + 1):
            self.layers[i].batch_size = batch_size

    def compute_misclass_softmax(self, X=None, y=None):
        """
        Compute the number of misclassifications for the sofmax layer(s).

        Parameters
        ----------
        X : array, optional
            Feature array. The default is None, in which case the entire training set is used.
        y : array, optional
            Target data array. The default is None, in which case the entire
            training set is used.

        Returns
        -------
        float
            The misclassification percentage in [0,1] per softmax layer.

        """

        n_misclass = np.zeros(self.n_softmax)

        # compute misclassification error of the training set if X and y are not set
        if y is None:
            print('Computing number of misclassifications wrt all training data.')
            X = self.X
            y = self.y
        else:
            print('Computing number of misclassifications wrt specified data (',
                  X.shape[0], 'samples)')

        n_samples = X.shape[0]

        for i in range(n_samples):
            _, max_idx_ann, _ = self.get_softmax(X[i].reshape([1, self.n_in]))

            max_idx_data = np.array([np.where(y_j == 1.0)[0]
                                     for y_j in np.split(y[i], self.n_softmax)])

            for j in range(self.n_softmax):
                if max_idx_ann[j] != max_idx_data[j]:
                    n_misclass[j] += 1

            if np.mod(i, 1000) == 0:
                print('Computing misclassification error:',
                      np.around(i / n_samples * 100, 1), '%')

        print('Number of misclassifications =', n_misclass)
        print('Misclassification percentage =', n_misclass / n_samples * 100, '%')

        return n_misclass / n_samples

    def get_n_weights(self):
        """
        Return the number of weights

        Returns
        -------
        n_weights : int
            The number of weights.

        """

        n_weights = 0

        for i in range(1, self.n_layers + 1):
            n_weights += self.layers[i].W.size

        print('This neural network has', n_weights, 'weights.')

        return n_weights

    def print_network_info(self):
        """
        Print some characteristics of the neural network to screen.

        Returns
        -------
        None.

        """
        print('===============================')
        print('Neural net parameters')
        print('===============================')
        print('Number of layers =', self.n_layers)
        print('Number of features =', self.n_in)
        print('Loss function =', self.loss)
        print('Number of neurons per hidden layer =', self.n_neurons)
        print('Number of output neurons =', self.n_out)
        print('Activation hidden layers =', self.activation)
        print('Activation output layer =', self.activation_out)
        # print('On GPU =', self.on_gpu)
        self.get_n_weights()
        print('===============================')
