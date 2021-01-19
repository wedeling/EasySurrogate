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
                 length_scale=1.0,
                 bias=True,
                 noize=True,
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

        self.kernel = Matern(length_scale=[length_scale]*self.n_in)

        if bias:
            self.kernel = self.kernel + ConstantKernel()

        if noize:
            self.kernel = self.kernel + WhiteKernel(noise_level=0.5)


        self.model = GaussianProcessRegressor(kernel=self.kernel, random_state=0)

    def train(self):
        self.model.fit(self.X)

    def forward(self, X_i):
        self.model.predict(X_i)

    def print_model_info(self):
        print('===============================')
        print('Gaussian Process parameters')
        print('===============================')
        print('Kernel =', self.kernel)
        print('Output dimensionality =', self.n_out)
        print('On GPU =', self.on_gpu)
        print('===============================')