import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog
from scipy.stats import rv_discrete


from .RNN_Layer import RNN_Layer
from .Input_Layer import Input_Layer


class RNN:

    def __init__(self, X, y, alpha=0.001, decay_rate=1.0,
                 decay_step=10**5, beta1=0.9, beta2=0.999, lamb=0.0,
                 n_out=1, param_specific_learn_rate=True, loss='squared',
                 activation='tanh', training_mode='offline', increment=1,
                 activation_out='linear', n_softmax=0, n_layers=2, n_neurons=16,
                 bias=True, sequence_size=100, save=True, load=False, name='ANN',
                 on_gpu=False, standardize_X=True, standardize_y=True, **kwargs):

        self.X = X
        self.y = y

        # number of input nodes
        try:
            self.n_in = X.shape[1]
        except IndexError:
            self.n_in = 1

        # number of training samples
        self.n_train = X.shape[0]

        # use either numpy or cupy via xp based on the on_gpu flag
        global xp
        if not on_gpu:
            import numpy as xp
        else:
            import cupy as xp

        self.on_gpu = on_gpu

        # standardize the training data
        if standardize_X:

            self.X_mean = xp.mean(X, axis=0)
            self.X_std = xp.std(X, axis=0)
            self.X = (X - self.X_mean) / self.X_std

        if standardize_y:
            self.y_mean = xp.mean(y, axis=0)
            self.y_std = xp.std(y, axis=0)
            self.y = (y - self.y_mean) / self.y_std

        # number of layers (hidden + output)
        self.n_layers = n_layers

        # number of layers in time
        self.sequence_size = sequence_size
        self.window_idx = 0
        self.max_windows_idx = self.n_train - sequence_size

        # number of neurons in a hidden layer
        self.n_neurons = n_neurons

        # number of output neurons
        self.n_out = n_out

        # use bias neurons
        self.bias = bias

        # number of bias neurons (0 or 1)
        self.n_bias = 0
        if self.bias:
            self.n_bias = 1

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

        self.training_mode = training_mode

        self.increment = increment

        # number of sofmax layers
        self.n_softmax = n_softmax

        # save the neural network after training
        self.save = save
        self.name = name

        self.loss_vals = []

        self.layers = []

        # input and hidden weight matrices
        self.W_in = []
        self.W_h = []

        self.test = []

        #####################################
        # Initialize shared weight matrices #
        #####################################

        # self.x_t = self.X[0, 0]

        # input weights
        self.W_in.append(
            xp.random.randn(
                self.n_in +
                self.n_bias,
                self.n_neurons) *
            xp.sqrt(
                1.0 /
                self.n_neurons))
        self.W_h.append(
            xp.random.randn(
                self.n_neurons +
                self.n_bias,
                self.n_neurons) *
            xp.sqrt(
                1.0 /
                self.n_neurons))

        # hidden weights
        for r in range(1, self.n_layers):
            self.W_in.append(
                xp.random.randn(
                    self.n_neurons +
                    self.n_bias,
                    self.n_neurons) *
                xp.sqrt(
                    1.0 /
                    self.n_neurons))
            self.W_h.append(
                xp.random.randn(
                    self.n_neurons +
                    self.n_bias,
                    self.n_neurons) *
                xp.sqrt(
                    1.0 /
                    self.n_neurons))

        # output weights (no W_h matrix in the output layer)
        self.W_in.append(
            xp.random.randn(
                self.n_neurons +
                self.n_bias,
                self.n_out) *
            xp.sqrt(
                1.0 /
                self.n_out))

        #########################
        # Create the RNN layers #
        #########################

        # add the input layer
        self.layers.append(Input_Layer(self.n_in, self.bias, self.sequence_size))

        # add the hidden layers
        for r in range(1, self.n_layers):
            self.layers.append(RNN_Layer(self.W_in[r - 1], self.W_h[r - 1],
                                         self.n_neurons, r, self.n_layers, 'tanh',
                                         self.loss, self.sequence_size,
                                         self.bias, lamb=lamb,
                                         on_gpu=on_gpu, n_softmax=self.n_softmax))

        # add the output layer
        self.layers.append(RNN_Layer(self.W_in[-1], None,
                                     self.n_out, r + 1, self.n_layers, 'linear',
                                     self.loss, self.sequence_size,
                                     False, lamb=lamb,
                                     on_gpu=on_gpu, n_softmax=self.n_softmax))

        self.connect_layers()

        self.print_network_info()

    # train the neural network

    def train(self, n_batch, store_loss=True):

        # number of epochs
        self.n_epoch = 0

        # initialize the history of the input layer
        self.layers[0].init_history()
        # initialize the first entries of the h_history and grad_Phi_history
        for r in range(1, self.n_layers):
            # initial previous h value
            h = self.layers[r].init_h_tm1()
            # never used, just a place filler
            grad_Phi = None
            self.layers[r].init_history(h, grad_Phi)

        for i in range(n_batch):

            # compute learning rate
            alpha = self.alpha * self.decay_rate**(np.int(i / self.decay_step))

            # run the batch
            self.batch(alpha=alpha, beta1=self.beta1, beta2=self.beta2)

            # store the loss value
            if store_loss:
                l = 0.0
                for k in range(self.n_out):
                    l += self.layers[-1].L_t

                if np.mod(i, 1000) == 0:
                    loss_i = xp.mean(l)
                    print('Batch', i, 'learning rate', alpha, 'loss:', loss_i)
                    # note: appending a cupy value to a list is inefficient - if done every iteration
                    # it will slow down executing significantly
                    self.loss_vals.append(loss_i)

        if self.save:
            self.save_ANN()

    # run the network forward

    def feed_forward(self, x_sequence=None, back_prop=False):
        """


        Parameters
        ----------
        x_sequence : array of inputs, size = (sequence size, size of 1 input)
            The default is None. If None, input sequences are obtained from
            the training data.
        back_prop : Bool
            DESCRIPTION. The default is False. Flag for performing back
            propagation.

        Returns
        -------
        y_hat_t :

        """

        # if no sequence is specified, get one from the training data
        if x_sequence is None and self.training_mode == 'offline':
            # generates the next sequence of continuous indices
            idx = self.slide_window(self.sequence_size, self.increment)
            x_sequence = self.X[idx, :]

            # predicted output sequence
            sequence_size = self.sequence_size
            # y_hat_t = xp.zeros([sequence_size, self.n_out])
        # #if training is done online
        # elif x_sequence is None and self.training_mode == 'online':
        #     #generates the next sequence of continuous indices
        #     idx = self.slide_window(self.sequence_size, self.sequence_size)

        #     x_t = self.x_t

        #     #predicted output sequence
        #     sequence_size = self.sequence_size
        #     y_hat_t = xp.zeros([sequence_size, self.n_out])

        else:
            assert x_sequence.ndim == 2
            sequence_size = x_sequence.shape[0]
            # y_hat_t = xp.zeros([sequence_size, self.n_out])

        # perform back propagation if True
        if back_prop:
            # data sequence to infer gradients from
            y_sequence = self.y[idx, :]

            # zero out the gradients of the previous sequence
            for r in range(1, self.n_layers):
                self.layers[r].L_grad_W_in = 0.0
                self.layers[r].L_grad_W_h = 0.0
            # zero out the gradient in the output layer
            self.layers[-1].L_grad_W_in = 0.0
            # set loss to zero
            self.layers[-1].L_t = 0.0

            # #if we are not at the beginning of the training data
            # if idx[0] != 0:
            #     #set the correct previous values of h and grad_Phis
            #     #for the next training sequence to ensure continuity
            #     self.layers[0].init_history()
            #     for r in range(1, self.n_layers):
            #         h = self.layers[r].h_history[self.increment]
            #         grad_Phi = self.layers[r].grad_Phi_history[self.increment]
            #         self.layers[r].init_history(h, grad_Phi)
            # else:
            #     self.clear_history()

        # count = 0

        # loop over all inputs of the input sequence
        for t in range(sequence_size):

            # current input
            if self.training_mode == 'offline':
                x_t = x_sequence[t].reshape([self.n_in, 1])

            # if a bias neuron is used, add 1 to the end of x_t
            if self.bias:
                x_t = xp.vstack([x_t, xp.ones(1)])

            input_feat = x_t
            for r in range(self.n_layers + 1):
                # compute the output, i.e. the hidden state, of the current layer
                hidden_state = self.layers[r].compute_output(input_feat)
                # hidden state becomes input for the next layer
                input_feat = hidden_state

            # final hidden state is the output
            # y_hat_t[t, :] = hidden_state.flatten()

        y_hat = hidden_state
        self.test.append(y_hat[0][0])
        # if self.training_mode == 'online':
        #     # x_t = 0.9*self.X[idx[count], 0] + 0.1*self.y[idx[count]]
        #     # x_t = 0.9*self.X[idx[count], 0] + 0.1*hidden_state
        #     x_t = 0.9*x_t[0:self.n_in] + 0.1*hidden_state
        #     count += 1

        if back_prop:
            # perform back propagation through time
            for t in range(self.sequence_size, 0, -1):

                if t == self.sequence_size:
                    # perform standard back propagation on the final time layer
                    self.back_prop(y_sequence[t - 1].reshape([self.n_out, 1]))
                else:
                    # perform back propagation through time
                    self.back_prop_through_time(y_sequence[t - 1].reshape([self.n_out, 1]), t)

        # self.x_t = x_t[0:self.n_in]

        return y_hat

    # back propagation algorithm

    def back_prop(self, y_t):

        # start back propagation over hidden layers and the output layer
        for r in range(self.n_layers, 0, -1):
            self.layers[r].back_prop(y_t)

    # back prop through time

    def back_prop_through_time(self, y_t, t):

        # start back propagation through time over only the hidden layers
        for r in range(self.n_layers, 0, -1):
            self.layers[r].back_prop_through_time(y_t, t)

    # get the output of the softmax layer (so far: only works for batch_size = 1)

    def get_softmax(self, X_i, feed_forward=True):

        if feed_forward:
            # feed forward features X_i
            h = self.feed_forward(x_sequence=X_i).flatten()
        else:
            h = self.layers[-1].h

        probs = []
        idx_max = []
        rvs = []

        for h_i in np.split(h, self.n_softmax):
            o_i = xp.exp(h_i) / xp.sum(np.exp(h_i), axis=0)
            o_i = o_i / np.sum(o_i)

            probs.append(o_i)

            idx_max.append(np.argmax(o_i))

            # Causing trouble: often gives ValueError: The sum of provided pk is not 1.
            # pmf = rv_discrete(values=(np.arange(o_i.size), o_i.flatten()))
            # rvs.append(pmf.rvs())

        # return values and index of highest probability and random samples from pmf
        return probs, idx_max, rvs

    def slide_window(self, sequence_size, increment):

        if self.window_idx > self.max_windows_idx:
            self.window_idx = 0
            self.n_epoch += 1
            print("Performed", self.n_epoch, "epochs.")

        idxs = range(self.window_idx, self.window_idx + sequence_size)
        self.window_idx += increment

        return idxs

    # update step of the weights

    def batch(self, alpha=0.001, beta1=0.9, beta2=0.999):

        self.feed_forward(back_prop=True)

        for r in range(1, self.n_layers + 1):

            layer_r = self.layers[r]

            # moving average of gradient and squared gradient magnitude
            layer_r.V_in = beta1 * layer_r.V_in + (1.0 - beta1) * layer_r.L_grad_W_in
            layer_r.A_in = beta2 * layer_r.A_in + (1.0 - beta2) * layer_r.L_grad_W_in**2

            # TODO: find better way, this if condition appears three times
            if r < self.n_layers:
                layer_r.V_h = beta1 * layer_r.V_h + (1.0 - beta1) * layer_r.L_grad_W_h
                layer_r.A_h = beta2 * layer_r.A_h + (1.0 - beta2) * layer_r.L_grad_W_h**2

            # select learning rate
            if not self.param_specific_learn_rate:
                # same alpha for all weights
                alpha_in = alpha
                alpha_h = alpha
            # param specific learning rate
            else:
                # RMSProp
                alpha_in = alpha / (xp.sqrt(layer_r.A_in + 1e-8))
                if r < self.n_layers:
                    alpha_h = alpha / (xp.sqrt(layer_r.A_h + 1e-8))

                # Adam
                #alpha_t = alpha*xp.sqrt(1.0 - beta2**t)/(1.0 - beta1**t)
                #alpha_i = alpha_t/(xp.sqrt(layer_r.A + 1e-8))

            # gradient descent update step
            if self.lamb > 0.0:
                # with L2 regularization
                layer_r.W = (1.0 - layer_r.Lamb * alpha_i) * layer_r.W - alpha_i * layer_r.V
            else:
                # without regularization
                layer_r.W_in = layer_r.W_in - alpha_in * layer_r.V_in
                if r < self.n_layers:
                    layer_r.W_h = layer_r.W_h - alpha_h * layer_r.V_h

                # Nesterov momentum
                # layer_r.W += -alpha*beta1*layer_r.V

    def clear_history(self):
        # clear input history
        self.layers[0].init_history()
        # Clear the hidden history of all layers
        for r in range(1, self.n_layers):
            # initial previous h value
            h = self.layers[r].init_h_tm1()
            # never used, just a place filler
            grad_Phi = None
            self.layers[r].init_history(h, grad_Phi)

    # connect each layer in the NN with its previous and the next

    def connect_layers(self):

        self.layers[0].meet_the_neighbors(None, self.layers[1])
        self.layers[-1].meet_the_neighbors(self.layers[-2], None)

        for i in range(1, self.n_layers):
            self.layers[i].meet_the_neighbors(self.layers[i - 1], self.layers[i + 1])

    # save using pickle (maybe too slow for very large RNNs?)

    def save_ANN(self, file_path=""):

        if len(file_path) == 0:

            root = tk.Tk()
            root.withdraw()

            file = filedialog.asksaveasfile(title="Save network",
                                            mode='wb', defaultextension=".pickle")
        else:
            file = open(file_path, 'wb')

        print('Saving ANN to', file.name)

        pickle.dump(self.__dict__, file)
        file.close()

    # load using pickle

    def load_ANN(self, file_path=""):

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

        self.print_network_info()

    # return the number of weights

    def get_n_weights(self):

        n_weights = 0

        # sum weight matrix sizes of all hidden layers (sizes W_in and W_h)
        for i in range(self.n_layers):
            n_weights += self.W_in[i].size + self.W_h[i].size

        # sum weight matrix sizes of output layers (size W_in only)
        n_weights += self.W_in[i].size

        print('This neural network has', n_weights, 'weights.')

        return n_weights

    def print_network_info(self):
        print('===============================')
        print('RNN parameters')
        print('===============================')
        print('Number of layers =', self.n_layers)
        print('Number of features =', self.n_in)
        print('Loss function =', self.loss)
        print('Number of neurons per hidden layer =', self.n_neurons)
        print('Number of output neurons =', self.n_out)
        print('Activation hidden layers =', self.activation)
        print('Activation output layer =', self.activation_out)
        print('On GPU =', self.on_gpu)
        self.get_n_weights()
        print('===============================')
