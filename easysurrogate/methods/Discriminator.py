"""
Discriminator class
"""

import numpy as np
from tqdm import tqdm

from .NN import ANN
from .GAN_Layer import GAN_Layer

class Discriminator(ANN):
    
    def __init__(self, X, alpha=0.001, decay_rate=1.0, decay_step=10**5, beta1=0.9,
                 beta2=0.999, lamb=0.0, activation='tanh',
                 n_layers=2, n_neurons=16,
                 bias=True, batch_size=1, param_specific_learn_rate=True,
                 save=False, on_gpu=False, name='Discriminator'):

        n_out = 1
        n_softmax = 0
        loss = 'discriminator_loss'
        activation_out = 'sigmoid'
        y = None

        super().__init__(X, y, alpha, decay_rate=decay_rate, 
                         decay_step=decay_step,
                         beta1=beta1, beta2=beta2, lamb=lamb, n_out=n_out, 
                         loss=loss, activation=activation,
                         activation_out=activation_out, n_softmax=n_softmax,
                         n_layers=n_layers, n_neurons=n_neurons,
                         bias=bias, batch_size=batch_size, 
                         param_specific_learn_rate=param_specific_learn_rate,
                         save=save, on_gpu=on_gpu, name=name,
                         standardize_X=False, 
                         standardize_y=False)

    def init_network(self, **kwargs):
        """
        Set up the network structure by creating the Layer objects and
        connecting them together.

        Returns
        -------
        None.

        """

        # add the inputs and hidden layers
        for r in range(self.n_layers):
            self.layers.append(
                GAN_Layer(
                    self.layer_sizes[r],
                    r,
                    self.n_layers,
                    self.layer_activation[r],
                    self.loss,
                    self.bias[r],
                    batch_size=self.batch_size,
                    lamb=self.lamb,
                    on_gpu=self.on_gpu))

        # add the output layer
        self.layers.append(
            GAN_Layer(
                self.n_out,
                self.n_layers,
                self.n_layers,
                self.activation_out,
                self.loss,
                bias=False,
                batch_size=self.batch_size,
                lamb=self.lamb,
                n_softmax=self.n_softmax,
                on_gpu=self.on_gpu,
                **kwargs))

    def train(
            self,
            n_batch,
            verbose=True,
            dropout=False, **kwargs):
        """
        Train the neural network using stochastic gradient descent.

        Parameters
        ----------
        n_batch : int
            The number of mini-batch iterations.
        verbose : boolean, optional
            Print information to screen while training. The default is True.
        dropout : boolean, optional
            Use dropout regularization. The default is False. To manually
            specify the dropout probabilities, specify the keyword argument
            "dropout_prob", as a list of probabilities of retaining neurons
            per layer. Otherwise, 0.8 is used for the input layer, 
            and 0.5 for the hidden layers.

        Returns
        -------
        None.

        """

        if dropout:
            self.dropout = dropout
            # use standard dropout probabilities
            if 'dropout_prob' not in kwargs:
                self.dropout_prob = [0.8]
                for i in range(self.n_layers - 1):
                    self.dropout_prob.append(0.5)
            # user-specified dropout probabilities
            else:
                self.dropout_prob = kwargs['dropout_prob']

        # loop with tqdm progress bar
        for i in tqdm(range(n_batch)):

            # select a random training instance (X, y)
            rand_idx = np.random.randint(0, self.n_train, self.batch_size)

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
            l = self.layers[-1].L_i
            loss_i = np.mean(l)
            self.loss_vals.append(loss_i)

            if np.mod(i, 1000) == 0:
                if verbose:
                    tqdm.write(' loss = %.4f' % (loss_i,))

        if self.dropout:
            # scale all weight matrices by dropout prob after training
            for i in range(1, self.n_layers + 1):
                self.layers[1].W *= self.dropout_prob[i - 1]

            # turn off dropout after training
            self.dropout = False

        if self.save:
            self.save_ANN()

    def batch(self, X_i, G_i, alpha=0.001, beta1=0.9, beta2=0.999, **kwargs):
        
        #TODO: adapt this to GANs
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

            # gradient descent update step with L2 regularization
            if self.lamb > 0.0:
                layer_r.W = (1.0 - layer_r.Lamb * alpha_i) * layer_r.W - alpha_i * layer_r.V
            # without regularization
            else:
                layer_r.W = layer_r.W - alpha_i * layer_r.V


    def get_softmax(self, X_i):
        raise AttributeError("Discriminator has no softmax output.")

    def compute_misclass_softmax(self, X=None, y=None):
        raise AttributeError("Discriminator has no softmax output.")
