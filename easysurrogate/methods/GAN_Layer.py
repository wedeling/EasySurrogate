"""
Layer class for GANs
"""

import sys
import numpy as np

from .Layer import Layer

class GAN_Layer(Layer):
    """
    Layer class for GANs    
    """

    def __init__(self, n_neurons, r, n_layers, activation, loss, bias=False,
                 batch_size=1, lamb=0.0, on_gpu=False, **kwargs):

        super().__init__(n_neurons, r, n_layers, activation, loss, bias=bias,
                         batch_size=batch_size, lamb=lamb, 
                         on_gpu=on_gpu, **kwargs)

    def compute_loss(self, D_x, D_z):
        
        # only compute if in an output layer
        if self.layer_rp1 is None:
            
            if self.loss == 'discriminator_loss':
                self.L_i = np.log(D_x) + np.log(1 - D_z)
            elif self.loss == 'generator_loss':
                self.L_i = np.log(D_z)
            else:
                print('Cannot compute loss: unknown loss and/or activation function')
                sys.exit()

    def compute_delta_oo(self, D_x, D_z):

        # if the neuron is in the output layer, initialze delta_oo
        if self.layer_rp1 is None:

            # compute the loss function
            self.compute_loss(D_x, D_z)
            
            if self.loss == 'discriminator_loss':
                
                self.delta_ho = 1 / D_x - 1 / (1 - D_z)
            
            elif self.loss == 'generator_loss':

                self.delta_ho = 1 / D_z