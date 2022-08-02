"""
Class and functions for Gaussian Process Regression model
------------------------------------------------------------------------------
Author: Y. Yudin
==============================================================================
"""

import numpy as np
from itertools import product
from scipy.optimize import minimize


class GaussianProcess():

    def __init__(self, **kwargs):
        """
        Initializing default attributes of the object
        """
        self.n_x_dim = 1
        
        kernel_name = 'sq_exp'
        if 'kernel' in kwargs:
            kernel_name = kwargs['kernel']

        self.set_kernel(kernel_name)

    def set_kernel(self, kernel='sq_exp'):
        """
        Setting type of the kernel function and its default parameters
        """

        if kernel == 'gibbs':
            self.kernel = gibbs_ns_kernel
        elif kernel == 'sq_exp':
            self.kernel = sq_exp_kernel_function
        else:
            self.kernel = sq_exp_kernel_function

        # default values of kernel parameters
        self.sigma_f = 1.0
        self.sigma_n = 0.1
        self.l = 1.0

    def set_covariance(self, K):

        self.K = K
        self.n = K.shape[0]

    def calc_covariance(self, X, kernel, sigma_f, l, n):
        """
        Returns:
             K: array_like
             covaraince defined element-wise as kernel function of X
        """

        K = [kernel(i, j, sigma_f=sigma_f, l=l) for (i, j) in product(X, X)]

        K = np.array(K).reshape(n, n)

        return K

    def fit_cov(self, X, covariance='regular'):

        #K = [self.kernel(i, j, sigma_f=self.sigma_f, l=self.l) for (i, j) in product(X, X)]
        #K = np.array(K).reshape(self.n, self.n)

        K = self.calc_covariance(X, self.kernel, self.sigma_f, self.l, self.n)
        self.set_covariance(K)

        K_inv_tot = np.linalg.inv(self.K + (self.sigma_n ** 2) * np.eye(self.n))
        self.K_inv_tot = K_inv_tot

    def optmize_hyperparameters(self, X_train, y_train):
        """
        Optimizes hyperparamter values for minimum of R^2 score on training dataset
        #TODO has to optimise from MLE or MAP
        Uses scipy .minimize() to find the optimum
        Reassigns the attributes of the object after optimization
        """

        self.X_train = X_train
        self.y_train = y_train

        hp_curval = np.array([self.sigma_f, self.sigma_n, self.l])

        hp_optval_res = minimize(self.r2_score_hp, hp_curval, options={'maxiter': 100})

        hp_optval = hp_optval_res.x

        self.sigma_f = hp_optval[0]
        self.sigma_n = hp_optval[1]
        self.l = hp_optval[2]

    def fit(self, X, y):

        self.n = X.shape[0]
        self.X = X
        self.y = y

        self.fit_cov(X)

        self.optmize_hyperparameters(X, y)

    def predict_mean(self, X_new):

        # Size of the new sample
        #print('X_new.shape={0}'.format(X_new.shape)) ###DEBUG

        if len(X_new.shape) == 1:
            X_new = np.array(X_new).reshape((1, X_new.shape[0]))
            n_star = 1
        else:   
            n_star = X_new.shape[0]
        
        #print('X_new.shape={0}'.format(X_new.shape)) ###DEBUG
        #print('n_star={0}'.format(n_star)) ###DEBUG
        #print('X.shape={0}'.format(self.X.shape)) ###DEBUG

        # Covariance matrix of new and old sample K_*=K(X_new,X) e R^(n x N)
        K_star = [
           self.kernel(
                i,
                j,
                sigma_f=self.sigma_f,
                l=self.l) for (
                i,
                j) in product(
                X_new,
                self.X)
                 ]
        
        K_star = np.array(K_star).reshape((n_star, self.n))

        # TODO: y after reshaping gives wrong output dimensionality
        #print('self.y = {0} ; self.n = {1} ; self.n_x_dim = {2} ; n_star = {3}'.
        #       format(self.y, self.n, self.n_x_dim, n_star)) ###DEBUG

        f_bar_star = np.dot(K_star, np.dot(self.K_inv_tot, self.y.reshape(self.n, self.n_x_dim)))

        #print('X_new.shape = {1} ; y.shape = {2} ; f_bar_star.shape = {0} ; K_star.shape = {3}, K_inv_tot.shape = {4} \n'.
        #       format(f_bar_star.shape, X_new.shape, self.y.shape, K_star.shape, self.K_inv_tot.shape)) ### DEBUG
        
        #print(' X_new = {1} \n y = {2} \n f_bar_star = {0} \n'.format(f_bar_star, X_new, self.y)) ### DEBUG
        
        return f_bar_star

    def predict_var(self, X_new):

        # Size of the new sample
        if len(X_new.shape) == 1:
            X_new = np.array(X_new).reshape((1, X_new.shape[0]))
            n_star = 1
        else:   
            n_star = X_new.shape[0]

        K_star2 = [
            self.kernel(
                i,
                j,
                sigma_f=self.sigma_f,
                l=self.l) for (
                i,
                j) in product(
                X_new,
                X_new)]

        K_star2 = np.array(K_star2).reshape((n_star, n_star))

        K_star = [
            self.kernel(
                i,
                j,
                sigma_f=self.sigma_f,
                l=self.l) for (
                i,
                j) in product(
                X_new,
                self.X)]

        K_star = np.array(K_star).reshape((n_star, self.n))

        cov_f_star = K_star2 - np.dot(K_star, np.dot(self.K_inv_tot, K_star.T))
        var_f_star = np.diag(cov_f_star)

        return var_f_star

    def predict(self, X, return_std=True):
        """
        Returns
        -------
            y_mean: array_like
            An array of values withe the same length as X meaning the mean of the p(y|X) posterior of the regression model
        """

        y_mean = self.predict_mean(X)

        if return_std:
            y_var = self.predict_var(X)
            return y_mean, y_var

        return y_mean, y_var

    def r2_score(self, X, y):
        """
        Returns
        -------
            r2: float
            A real value r e [0.;1.] with 1. meaning that regression model fully captures data variance
        """
        f = self.predict_mean(X)
        y_mean = y.mean()
        r2 = 1 - np.multiply(y - f, y - f).sum() / (np.multiply(y - y_mean, y - y_mean).sum())
        return r2

    def r2_score_hp(self, hpval):
        """
        Calculates R2 variance explanation coefficent.
        Function signature complies with scipy.optimize.minimize() 
        """

        self.sigma_f = hpval[0]
        self.sigma_n = hpval[1]
        self.l = hpval[2]
        self.fit_cov(self.X_train)
        return self.r2_score(self.X_train, self.y_train)

    def score(self, X, y):
        """
        Synonim for r2_score
        """
        return self.r2_score(X, y)

    def calc_H(self, X, h=lambda x:x):
        """
        Calculate matrix of basis vectors for the model of form: y = h(x)*beta+f(x)
        """

        n = len(X)
        #H = [h(x) for x in list(X)]
        #H = np.array(H).reshape((n,n))
        H = np.ones((n,n))
        return H

    def beta_of_theta(self, y, sigma_n, sigma_f, l):
        """
        Estimate optimal beta (coefficient vector for y = h(x)*beta+f(x)) for given parameters of kernel theata: (sigma_f, l) 
        """

        n = len(y)

        K = self.calc_covariance(self.X, sigma_f, l, n)
        K_modif = K + sigma_n**2 * np.eye(n)

        H = self.calc_H(self.X)
        
        beta_hat = np.linalg.inv(H * np.linalg.inv(K_modif) * H) \
                   * H.T * np.linalg.inv(K_modif) * y

        return beta_hat

    def marg_log_likelihood(self, y, X, beta, theta, sigma_n, h=lambda x:x):
        """
        Returns:
            mml: float
            p(y|X, kernel_params)
        """
        
        # y - observable QoI values
        # H - matrix of vector functions, after whitening could be identity functions
        # beta - model coefficient vector (in basis functions)
        # K - covariance matrix K(X,X|theta)
        # sigma_n - nugget value
        
        # theta - parameters of the kernel (sigma_f, l, [nu])
        # use logarithm with base 2
        # recalculate K based on X, and theta 
        # mind fast matrix inversion or decomposition

        n = len(y)

        K = self.calc_covariance(X, self.kernel, theta['sigma_f'], theta['l'], n)

        H = self.calc_H(self.X)

        # use this to reduce optimisation space for GPR MLE fitting
        beta_hat = self.beta_of_theta(y, sigma_n, theta['simga_f'], theta['l'])
        beta = beta_hat

        # Option for GPR
        mml = -0.5 * (y - H*beta).T * np.linalg.inv(K + sigma_n**2 * np.eye(n)) * (y - H*beta) \
              - n/2. * np.log2(2*np.pi) \
              - 0.5 * np.log2(np.abs(K + sigma_n**2 * np.eye(n)))

        # Option for STP
        """
        mml = np.log2(np.gamma((theta['nu'] + n) / 2.)) \
                - np.log2(np.gamma(theta['nu'] / 2.)) \
                - n / 2. * np.log2(theta['nu'] * np.pi) \
                - 0.5 * np.log2(((theta['nu'] - 2.) / theta['nu']) * K) \
                - ((theta['nu'] + n) / 2.) * np.log2(1. + ((y.T * np.linalg.inv(K) *y) / theta['nu']))
                # K should include nugget and be K + sigma_n**2 * I_n ?
        """

        # use to find argmax of MLE over {beta, theta, sigma_n} 

        return mml

########################################
### Kernel functions implementations ###
########################################

def gibbs_ns_kernel(x, y, l, l_func=lambda x: x):
    """
    Defines Gibbs kernel function
    """
    pref_val = 1.0
    exp_arg = 0.0
    for d in x.shape[1]:
        # multiplicatve prefactor
        pref_val *= np.sqrt((2 * l_func(x) * l_func(y)) / (l_func(x)**2 + l_func(y)**2))
        # exponential argument
        exp_arg += (x - y)**2 / (l_func(x)**2 + l_func(y)**2)
    exp_val = np.exp(-exp_arg)
    return pref_val * exp_val

def sq_exp_kernel_function(x, y, sigma_f=1., l=1.):
    """
    Defines squared exponential kernel function
    """
    kernel = sigma_f * np.exp(- (np.linalg.norm(x - y)**2) / (2 * l**2))
    return kernel
