import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, RBF, ConstantKernel
from sklearn.metrics import mean_squared_error as mse

import mogp_emulator as mogp
from mogp_emulator import GaussianProcess, MultiOutputGP
from mogp_emulator.MeanFunction import Coefficient, LinearMean, MeanFunction


class GP:

    def __init__(
            self,
            kernel='Matern',
            n_in=1,
            n_out=1,
            length_scale=1.0,
            prefactor=True,
            bias=False,
            noize=1e-8,
            n_iter=1,
            save=True,
            load=False,
            name='GP',
            on_gpu=False,
            backend='scikit-learn',
            standardize_X=True,
            standardize_y=True,
            **kwargs):

        self.n_in = n_in
        self.n_out = n_out

        self.backend = backend

        self.on_gpu = on_gpu

        self.kernel_argument = ''
        self.noize_argument = ''

        # sciki-learn specific part
        if self.backend == 'scikit-learn':
            self.kernel = ConstantKernel(constant_value=1.0,
                                         constant_value_bounds=(1e-6, 1e+6))

            if kernel == 'Matern':
                self.kernel *= Matern(length_scale=[length_scale] *
                                      self.n_in, length_scale_bounds=[length_scale *
                                                                      1e-4, length_scale *
                                                                      1e+4], nu=2.5)

            elif kernel == 'RBF':
                self.kernel *= RBF(length_scale=[length_scale] * self.n_in,
                                   length_scale_bounds=[length_scale * 1e-4, length_scale * 1e+4])

            if bias:
                self.kernel += ConstantKernel(constant_value=1.0,
                                              constant_value_bounds=(1e-5, 1e+5))

            noize_val = 1e-8
            bounds_val = (noize_val * 1e-3, noize_val * 1e+3)
            if noize == 'adaptive':
                noize_val = 1e-12
                bounds_val = 'fixed'
            elif isinstance(noize, float):
                noize_val = noize
                bounds_val = (noize_val * 1e-3, noize_val * 1e+3)

            if noize is not False:
                self.kernel += WhiteKernel(noise_level=noize_val,
                                           noise_level_bounds=bounds_val)

            self.n_iter = n_iter

        # MOGP specific part
        elif self.backend == 'mogp':

            if kernel == 'Matern':
                self.kernel_argument += 'Matern52'
            elif kernel == 'RBF':
                self.kernel_argument += 'SquaredExponential'

            if isinstance(noize, float):
                self.noize_argument = noize
            elif noize == 'fit' or noize is True:
                self.noize_argument = 'fit'
            elif noize == 'adaptive':
                self.noize_argument = 'adaptive'
            else:
                self.noize_argument = 'adaptive'  # redundant, but keeping a default option for future

            if bias:
                raise NotImplementedError('Non-stationary kernels are not implemented in MOGP')

            # self.mean = Coefficient() + Coefficient() * LinearMean()

        else:
            raise NotImplementedError('Currently supporting only scikit-learn and mogp backend')

    def train(self, X, y):

        self.n_train = X.shape[0]

        try:
            n_in = X.shape[1]
            if self.n_in != n_in:
                raise RuntimeError('Size of training data feature is different from expected')
        except IndexError:
            if self.n_in != 1:
                raise RuntimeError(
                    'Size of training data feature is different from expected default =1')

        try:
            n_out = y.shape[1]
            if self.n_out != n_out:
                raise RuntimeError('Size of training data target is different from expected')
        except IndexError:
            if self.n_out != 1:
                raise RuntimeError(
                    'Size of training data target is different from expected default =1')

        if self.backend == 'scikit-learn':
            self.instance = GaussianProcessRegressor(
                kernel=self.kernel, n_restarts_optimizer=self.n_iter, normalize_y=True)
            self.instance.fit(X, y)
            self.kernel = self.instance.kernel_
        elif self.backend == 'mogp':
            if self.n_out == 1:
                self.instance = GaussianProcess(X, y.reshape(-1), kernel=self.kernel_argument,
                                                nugget=self.noize_argument)
            else:
                self.instance = MultiOutputGP(X, y.T, kernel=self.kernel_argument,
                                              nugget=self.noize_argument)
            self.instance = mogp.fit_GP_MAP(self.instance)

    def predict(self, X_i):

        if self.backend == 'scikit-learn':
            # for single sample X_i should be nparray(1, n_feat)
            m, v = self.instance.predict(X_i.reshape(1, -1), return_std=True)
            d = np.zeros(m.shape)
        elif self.backend == 'mogp':
            m, v, d = self.instance.predict(X_i, unc=True, deriv=True)
        else:
            raise NotImplementedError('Non-stationary kernels are not implemented in MOGP')

        return m, v, d

    def forward(self, X_i):  # no cases when required different from .predict() for GP case
        m, v = self.instance.predict(X_i)
        return m  # for single sample should be nparray(1,n_feat)

    def print_model_info(self):
        print('===============================')
        print('Gaussian Process parameters')
        print('===============================')
        if self.backend == 'scikit-learn':
            # print('Kernel params =', self.instance.kernel_.get_params())
            print('Kernel =', self.instance.kernel_)
            print('Kernel theta =', self.instance.kernel_.theta)
            print('Surrogate model parameters :', self.instance.get_params())
        elif self.backend == 'mogp':
            if isinstance(self.instance, MultiOutputGP):
                print('Model parameter printed not implemented for MOGP vector QoI models')
            else:
                print('Kernel =', self.instance.kernel)
                print('Kernel theta =', self.instance.theta)
        print('Output dimensionality =', self.n_out)
        print('Input dimensionality =', self.n_in)
        print('On GPU =', self.on_gpu)
        print('===============================')
