"""
Batch normalization, applies a transformation that maintains the mean
layer output close to 0 and the standard deviation close to 1.


Reference:
---------
Ioffe, Sergey, and Christian Szegedy.
Batch normalization: Accelerating deep network training by reducing
internal covariate shift." International conference on machine learning.
pmlr, 2015.

"""

import numpy as np


class Batch_Normalization:
    """
    Class for Batch Normalization
    """

    def __init__(self, layer, momentum=0.99, epsilon=0.001):
        """
        Initialize a BatchNormalization object.

        Parameters
        ----------
        layer : EasySurrogate Layer object
            The layer of a neural network to which Batch Normalization
            must be applied.
        momentum : float, optional
            Parameter of the moving average, applied to the batch mean and
            variance. The default is 0.99.
        epsilon : float, optional
            Small parameter added to the batch variance, to prevent divide
            by zero during normalization. The default is 0.001.

        Returns
        -------
        None.

        """
        self.layer = layer
        self.epsilon = epsilon
        self.momentum = momentum
        self.training = False
        self.mean = 0
        self.var = 1
        self.std = 1
        self.moving_mean = 0
        self.moving_std = 1
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights used in Batch Normalization, namely
        beta and gamma and associated mometum & squared gradient matrices.

        Beta is in itialized to zero, and gamma to one.

        Returns
        -------
        None.

        """
        # the bias of the normalized pre-activation output.
        self.beta = np.zeros([self.layer.n_neurons, 1])
        # the scaling factor of the normalized pre-activation output.
        self.gamma = np.ones([self.layer.n_neurons, 1])

        # momentum beta
        self.V_beta = np.zeros(self.beta.shape)
        # squared gradient beta
        self.A_beta = np.zeros(self.beta.shape)
        # momentum gamma
        self.V_gamma = np.zeros(self.gamma.shape)
        # squared gradient gamma
        self.A_gamma = np.zeros(self.gamma.shape)

    def set_training(self, training):
        """
        Set the training flag of batch normalization. If true, the pre-activation
        is normalized using the batch mean and standard deviation. If False,
        the running averages of the mean and standard deviation computed during
        training are used instead.

        Parameters
        ----------
        training : bool
            The training flag.

        Returns
        -------
        None.

        """
        assert isinstance(training, bool), "Training flag must be boolean."
        self.training = training

    def normalize(self, v):
        """
        Normalize the pre-activation output of the layer.

        Parameters
        ----------
        v : array, shape (n_neurons, batch_size)
            The unnormalized pre-activation output of the layer.

        Returns
        -------
        a, array, shape (n_neurons, batch_size)
            The normalized pre-activation output of the layer.

        """
        # the linear preactivation output of the layer
        self.v = v

        # if the training flag is set, use the mini-batches to compute
        # the mean and variance
        if self.training and v.shape[1] > 1:

            # compute mean and variance of v over the mini batch
            self.mean = np.mean(self.v, axis=1, keepdims=True)
            self.var = np.var(self.v, axis=1, keepdims=True) + self.epsilon
            self.std = self.var ** 0.5

            # compute a moving average of the moments for inference
            self.moving_mean = self.moving_mean * self.momentum + self.mean * (1 - self.momentum)
            self.moving_std = self.moving_std * self.momentum + self.std * (1 - self.momentum)

            # standardize v
            self.v_hat = (self.v - self.mean) / self.std
            # linearly parameterize v_hat to allow for full expressive power
            # of activation function
            self.a = self.gamma * self.v_hat + self.beta
        # if the training flag is not set, use the moving averages for
        # normalization
        else:
            # standardize v using the computed moving averages
            self.v_hat = (self.v - self.moving_mean) / self.moving_std
            # linearly parameterize v_hat to allow for full expressive power
            self.a = self.gamma * self.v_hat + self.beta

        return self.a

    def compute_L_grad_W(self):
        """
        Compute the gradient of the loss function with respect to the weight
        of the layer. Propagates the loss through the batch-normalization
        procedure to compute the loss gradient wrt the unnormalized
        pre-activation output v. This gradient is used to compute dL/dW.
        The loss gradients wrt beta and gamma are also computed here.

        Returns
        -------
        L_grad_W : array, shape (n_neurons + n_bias previous layer, n_neurons this layer)
            The loss gradient.

        """

        # keyword arguments to use in all np.sum() calls
        kwargs = {'axis': 1, 'keepdims': True}

        # the gradient loss wrt a, the pre activation output after
        # the batch norm mode
        self.dL_da = self.layer.delta_ho * self.layer.grad_Phi

        # the gradient loss of the parameters of a
        self.L_grad_gamma = np.sum(self.dL_da * self.v_hat, **kwargs)
        self.L_grad_beta = np.sum(self.dL_da, **kwargs)

        # the gradient loss wrt the mini batch variance
        v_minus_mean = self.v - self.mean
        self.dL_dvar = np.sum(v_minus_mean * self.dL_da, **kwargs)
        self.dL_dvar *= -self.gamma / (2 * self.std ** 3)

        # size of the mini batch
        M = self.layer.batch_size

        # the gradient loss wrt the mini batch mean
        self.dL_dmean = self.gamma / self.std ** 3 * \
            np.sum(v_minus_mean * self.dL_da, **kwargs) * \
            np.sum(v_minus_mean / M, **kwargs) - \
            self.gamma / self.std * np.sum(self.dL_da, **kwargs)

        # the gradient loss wrt preactivation output before the
        # batch norm node
        self.dL_dv = self.gamma / self.std * self.dL_da + self.dL_dmean / M + \
            2 * v_minus_mean * self.dL_dvar / M

        # computer the loss gradient wrt the weights of this layer
        h_rm1 = self.layer.layer_rm1.h
        L_grad_W = np.dot(h_rm1, self.dL_dv.T) / M

        return L_grad_W
