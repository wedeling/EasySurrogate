import numpy as np
import pickle

from .DAS_Layer import DAS_Layer
from .Layer import Layer
from .NN import ANN

class DAS_surrogate(ANN):

    def __init__(
            self,
            X,
            y,
            d,
            alpha=0.001,
            decay_rate=1.0,
            decay_step=10**5,
            beta1=0.9,
            beta2=0.999,
            lamb=0.0,
            n_out=1,
            param_specific_learn_rate=True,
            loss='squared',
            activation='tanh',
            activation_out='linear',
            n_softmax=0,
            n_layers=2,
            n_neurons=16,
            bias=True,
            neuron_based_compute=False,
            batch_size=1,
            save=True,
            load=False,
            name='DAS',
            on_gpu=False,
            standardize_X=True,
            standardize_y=True,
            **kwargs):

        # the features
        self.X = X

        # number of training data points
        self.n_train = X.shape[0]

        # the training outputs
        self.y = y
        
        #the dimension of the active subspace
        self.d = d

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

        # determines where to compute the neuron outputs and gradients
        # True: locally at the neuron, False: on the Layer level in one shot via linear algebra)
        self.neuron_based_compute = neuron_based_compute

        # size of the mini batch used in stochastic gradient descent
        self.batch_size = batch_size

        self.loss_vals = []

        self.layers = []

        # add the input layer
        self.layers.append(Layer(self.n_in, 0, self.n_layers, 'linear',
                                 self.loss, False, batch_size=batch_size, lamb=lamb,
                                 neuron_based_compute=neuron_based_compute, on_gpu=on_gpu))

        # add the deep active subspace layer
        self.layers.append(DAS_Layer(self.d, batch_size=batch_size))

        # add the hidden layers
        for r in range(2, self.n_layers):
            self.layers.append(Layer(self.n_neurons, r, self.n_layers, self.activation,
                                     self.loss, self.bias, batch_size=batch_size, lamb=lamb,
                                     neuron_based_compute=neuron_based_compute, on_gpu=on_gpu))

        # add the output layer
        self.layers.append(
            Layer(
                self.n_out,
                self.n_layers,
                self.n_layers,
                self.activation_out,
                self.loss,
                batch_size=batch_size,
                lamb=lamb,
                n_softmax=n_softmax,
                neuron_based_compute=neuron_based_compute,
                on_gpu=on_gpu,
                **kwargs))

        super().connect_layers()
        super().print_network_info()
        
    # run the network forward
    # X_i needs to have shape [batch size, number of features]
    def feed_forward(self, X_i, batch_size=1):

        # set the features at the output of in the input layer
        self.layers[0].h = X_i.T

        for i in range(1, self.n_layers + 1):
            # compute the output on the layer using matrix-maxtrix multiplication
            self.layers[i].compute_output(batch_size)

        return self.layers[-1].h

    # update step of the weights
    def batch(self, X_i, y_i, alpha=0.001, beta1=0.9, beta2=0.999, t=0):

        self.feed_forward(X_i, self.batch_size)
        self.back_prop(y_i)

        for r in range(1, self.n_layers + 1):

            layer_r = self.layers[r]

            #Deep active subspace layer
            if r == 1:
                # momentum
                layer_r.V = beta1 * layer_r.V + (1.0 - beta1) * layer_r.L_grad_Q
                # moving average of squared gradient magnitude
                layer_r.A = beta2 * layer_r.A + (1.0 - beta2) * layer_r.L_grad_Q**2
            #standard layer
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
                #Deep active subspace layer
                if r == 1:
                    #update the Q weights
                    layer_r.Q = layer_r.Q - alpha_i * layer_r.V
                    #compute the weights W(Q) via Gram Schmidt
                    layer_r.compute_weights()
                #standard layer
                else:
                    layer_r.W = layer_r.W - alpha_i * layer_r.V