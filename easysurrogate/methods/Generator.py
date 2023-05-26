"""
Generator class
"""

import numpy as np

from .NN import ANN

class Generator(ANN):
    
    def __init__(self, n_out, alpha=0.001, decay_rate=1.0, decay_step=10**5, beta1=0.9,
                 beta2=0.999, lamb=0.0, activation='tanh', activation_out = 'linear',
                 n_layers=2, n_neurons=16,
                 bias=True, batch_size=1, param_specific_learn_rate=True,
                 save=False, on_gpu=False, name='Generator'):

        n_softmax = 0
        loss = 'gan_minimax'
        X = np.random.randn(1, 1)
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
        
    def get_softmax(self, X_i):
        raise AttributeError("Generator has no softmax output.")

    def compute_misclass_softmax(self, X=None, y=None):
        raise AttributeError("Generator has no softmax output.")
