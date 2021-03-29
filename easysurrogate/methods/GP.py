import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog
from scipy.stats import rv_discrete
import h5py

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, RBF, ConstantKernel
from sklearn.metrics import mean_squared_error as mse


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
                 noize=True,
                 save=True,
                 load=False,
                 name='GP',
                 on_gpu=False,
                 standardize_X=True,
                 standardize_y=True,
                 **kwargs):

        #self.X = X
        #self.y = y

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

        self.kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-6, 1e+6))

        if kernel == 'Matern':
            #self.kernel = Matern(length_scale=[length_scale]*self.n_in)
            self.kernel *= Matern()
        elif kernel == 'RBF':
            self.kernel *= RBF(length_scale=[length_scale]*self.n_in, length_scale_bounds=[1e-4, 1e+4])

        if bias:
            self.kernel += ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-5, 1e+5))

        if noize:
            self.kernel += WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-09, 1e+2))

        self.instance = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, normalize_y=True)  #, random_state=42

        self.train(X, y)

    def train(self, X, y):
        self.instance.fit(X, y)

    def predict(self, X_i):
        m, v = self.instance.predict(X_i, return_std=True)  # for single sample X_i should be nparray(1,n_feat)
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