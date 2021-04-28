import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog
from scipy.stats import rv_discrete
import h5py

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, RBF, ConstantKernel
from sklearn.metrics import mean_squared_error as mse

import mogp_emulator as mogp
from mogp_emulator import GaussianProcess


class GP:

    def __init__(
                 self,
                 X,
                 y,
                 n_out=1,
                 kernel='Matern',
                 length_scale=1.0,
                 prefactor=True,
                 bias=False,
                 noize=1e-8,
                 save=True,
                 load=False,
                 name='GP',
                 on_gpu=False,
                 backend='scikit-learn',
                 standardize_X=True,
                 standardize_y=True,
                 **kwargs):

        self.backend = backend
        self.n_train = X.shape[0]

        try:
            self.n_in = X.shape[1]
        except IndexError:
            self.n_in = 1

        try:
            self.n_out = y.shape[1]
        except IndexError:
            self.n_out = 1

        self.on_gpu = on_gpu

        # sciki-learn specific part
        if self.backend == 'scikit-learn':
            self.kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e+6))

            if kernel == 'Matern':
                #self.kernel = Matern(length_scale=[length_scale]*self.n_in)
                self.kernel *= Matern()
            elif kernel == 'RBF':
                self.kernel *= RBF(length_scale=[length_scale]*self.n_in, length_scale_bounds=[1e-4, 1e+4])

            if bias:
                self.kernel += ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-5, 1e+5))

            if noize is not False:
                self.kernel += WhiteKernel(noise_level=noize, noise_level_bounds=(noize*1e-3, noize*1e+3))

            self.instance = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, normalize_y=True)  #, random_state=42

        # MOGP specific part
        if self.backend == 'mogp':

            y = y.reshape(-1)

            kernel_argument = ''

            if kernel == 'Matern':
                kernel_argument += 'Matern52'

            if kernel == 'RBF':
                kernel_argument += 'SquaredExponential'

            if noize is not False:
                noize_argument = noize

            if bias:
                raise NotImplementedError('Non-stationary kernels are not implemented in MOGP')

            self.instance = GaussianProcess(X, y, kernel=kernel_argument, nugget=noize_argument)

        else:
            raise NotImplementedError('Currently supporting only scikit-learn and mogp backend')

        self.train(X, y)

    def train(self, X, y):
        if self.backend == 'scikit-learn':
            self.instance.fit(X, y)
        if self.backend == 'mogp':
            self.instance = mogp.fit_GP_MAP(self.instance)

    def predict(self, X_i):
        if self.backend == 'scikit-learn':
            # for single sample X_i should be nparray(1, n_feat)
            m, v = self.instance.predict(X_i, return_std=True)
        if self.backend == 'mogp':
            m, v, d = self.instance.predict(X_i)
        else:
            raise NotImplementedError('Non-stationary kernels are not implemented in MOGP')
        return m, v

    def forward(self, X_i):  # for no cases when required different from predict at GP case
        m, v = self.instance.predict(X_i)
        return m  # for single sample should be nparray(1,n_feat)

    def print_model_info(self):
        print('===============================')
        print('Gaussian Process parameters')
        print('===============================')
        #print('Kernel =', self.instance.kernel)
        #print('Kernel params =', self.instance.kernel.get_params())
        #print('Kernel theta =', self.instance.kernel.theta)
        print('Output dimensionality =', self.n_out)
        print('Input dimensionality =', self.n_in)
        print('On GPU =', self.on_gpu)
        print('===============================')