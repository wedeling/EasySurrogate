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
                 bias=True,
                 noize=False,
                 save=True,
                 load=False,
                 name='GP',
                 on_gpu=False,
                 standardize_X=True,
                 standardize_y=True,
                 **kwargs):

        self.X = X

        self.n_train = X.shape[0]

        self.y = y

        try:
            self.n_in = X.shape[1]
        except IndexError:
            self.n_in = 1

        try:
            self.n_out = y.shape[1]
        except IndexError:
            self.n_out = 1

        if kernel == 'Matern':
            self.kernel = Matern(length_scale=[length_scale]*self.n_in)

        if bias:
            self.kernel = self.kernel + ConstantKernel()

        if noize:
            self.kernel = self.kernel + WhiteKernel(noise_level=0.5)

        self.instance = GaussianProcessRegressor(kernel=self.kernel, random_state=0)
        self.train()

    def train(self):
        self.instance.fit(self.X, self.y)

    def predict(self, X_i):
        m, v = self.instance.predict(X_i, return_std=True)  # for single sample X_i should be nparray(1,4)
        return m, v

    def forward(self, X_i):  # for no cases when required different from predict at GP case
        m, v = self.instance.predict(X_i)
        return m  # for single sample should be nparray(1,4)

    def print_model_info(self):
        print('===============================')
        print('Gaussian Process parameters')
        print('===============================')
        print('Kernel =', self.kernel)
        print('Output dimensionality =', self.n_out)
        print('On GPU =', self.on_gpu)
        print('===============================')