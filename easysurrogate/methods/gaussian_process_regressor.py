"""
Class and functions for Gaussian Process Regression model
------------------------------------------------------------------------------
Author: Y. Yudin
==============================================================================
"""

import numpy as np
from itertools import product
from scipy.optimize import minimize

from scipy.special import gamma, kv
from scipy.linalg import solve_triangular

class GaussianProcessRegressor():

    def __init__(self, **kwargs):
        """
        Initializing default attributes of the object
        """

        # Dimensionality of output
        self.n_y_dim = 1

        # Dimensionality of input
        if 'n_x_dim' not in kwargs:
            self.n = 1
        else:
            self.n = kwargs['n_x_dim']
        
        # Kernel type
        if 'kernel' not in kwargs:
            self.kernel_name = 'sq_exp'
        else:
            self.kernel_name = kwargs['kernel']
        self.set_kernel(self.kernel_name)

        # Default values of kernel parameters
        if 'sigma_f' not in kwargs:
            self.sigma_f = 1.0
        else:
            self.sigma_f = kwargs['sigma_f']

        if 'sigma_n' not in kwargs:
            self.sigma_n = 1.0
        else:
            self.sigma_n = kwargs['sigma_n']

        if 'l' not in kwargs:
            self.l = [1.]*self.n
        else:
            self.l = kwargs['l']

        if 'nu' not in kwargs:
            self.nu = 3
        else:
            self.nu = kwargs['nu']

        # Type of the stochastic process
        if 'process_type' not in kwargs:
            self.process_type = 'gaussian'
        else:
            self.process_type = kwargs['process_type']

    def set_kernel(self, kernel='sq_exp', **kwargs):
        """
        Setting type of the kernel function and its default parameters
        """

        if kernel == 'gibbs':
            self.kernel = gibbs_ns_kernel
        elif kernel == 'matern':
            self.kernel = matern_kernel
        else:
            self.kernel = sq_exp_kernel_function

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
        
        print('K[0][0]={0}, X[0]={1}, sigma_f={2}, l={3}'.format(K[0], X[0], sigma_f, l)) ###DEBUG
        #print('k(X[0], X[0])={0} \n'.format(kernel(X[0], X[0], sigma_f, l))) ###DEBUG

        K = np.array(K).reshape(n, n)

        return K

    def calc_noise(self, X_train_var):
        """
        Returns
        -------
            Eta: array_like
            Eta e R^[n x n] matrix of original X noise with X_var at the diagonal
        """

        Eta = np.diag(X_train_var) 
        #TODO could be expressed using an additional kernel -> increases the search parameter space

        return Eta

    def fit_cov(self, X, X_var=False, covariance='regular'):

        #K = [self.kernel(i, j, sigma_f=self.sigma_f, l=self.l) for (i, j) in product(X, X)]
        #K = np.array(K).reshape(self.n, self.n)

        K = self.calc_covariance(X, self.kernel, self.sigma_f, self.l, self.n)
        self.set_covariance(K)

        if not X_var:
            K_inv_tot = np.linalg.inv(self.K + (self.sigma_n ** 2) * np.eye(self.n))
        else:
            Eta = self.calc_noise(X_var)
            K_inv_tot = np.linalg.inv(self.K + Eta)
        
        self.K_inv_tot = K_inv_tot

    def optmize_hyperparameters(self, X_train, y_train, X_train_var=False, loss='r2'):
        """
        Optimizes hyperparamter values for minimum of R^2 score on training dataset OR perform MLE
        #TODO has to optimise from MLE or MAP -> MLe done
        Uses scipy .minimize() to find the optimum
        Reassigns the attributes of the object after optimization; assumes the covariance matrix is calculated before the first iteration
        """

        self.X_train = X_train
        self.y_train = y_train
        self.X_train_var = X_train_var

        hp_curval = np.array([self.sigma_f, self.sigma_n, self.nu, *self.l,])

        hp_bounds = [(1e-16, None), (1e-16, None), (1, None), *([(1e-16, None)]*len(self.l))]

        hp_options = {'maxiter': 100}

        if loss == 'nmll':
            loss_func = self.nmll_hp
        else:
            loss_func == self.r2_score_hp

        hp_optval_res = minimize(loss_func, hp_curval, bounds=hp_bounds, options=hp_options)

        hp_optval = hp_optval_res.x

        self.sigma_f = hp_optval[0]
        self.sigma_n = hp_optval[1]     
        self.nu = hp_optval[2]
        self.l = hp_optval[3:]  

        self.fit_cov(X_train, X_train_var)  

    def fit(self, X, y, X_var=False):
        """
        Sets the training data and associated attibutes.
        Calculates the covariance matrix and calls for optimisation of hyperparameters
        Name complies with scikit-learn and other packages standard .fit() method
        """

        self.n = X.shape[0]
        self.X = X
        self.y = y

        self.fit_cov(X, X_var)

        self.optmize_hyperparameters(X, y, X_var, loss='nmll')

    def predict_mean(self, X_new):

        # Size of the new sample

        if len(X_new.shape) == 1:
            X_new = np.array(X_new).reshape((1, X_new.shape[0]))
            n_star = 1
        else:   
            n_star = X_new.shape[0]

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

        f_bar_star = np.dot(K_star, np.dot(self.K_inv_tot, self.y.reshape(self.n, self.n_y_dim)))
        
        return f_bar_star

    def predict_var(self, X_new, likelihood='gaussian'):

        likelihood = self.process_type

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

        M1 = K_star2 - np.dot(K_star, np.dot(self.K_inv_tot, K_star.T))

        if likelihood == 'gaussian':
            
            cov_f_star = M1

        elif likelihood == 'student_t':
            
            M2 = np.dot(self.y.T, np.dot(self.K_inv_tot, self.y))
            cov_f_star = np.dot((self.nu + M2 - 2)/(self.nu + self.n - 2), M1)

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

        return y_mean

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
        self.nu = hpval[2]
        self.l = hpval[3:]

        self.fit_cov(self.X_train, self.X_train_var)
        return self.r2_score(self.X_train, self.y_train)

    def score(self, X, y):
        """
        Synonym for r2_score
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

        K = self.calc_covariance(self.X, self.kernel, sigma_f, l, n)
        K_modif = K + sigma_n**2 * np.eye(n)
        # TODO add option for heteroschedastic noise

        H = self.calc_H(self.X)
        
        beta_hat = np.linalg.inv(H.dot(np.linalg.inv(K_modif).dot(H)))\
                   .dot(H.T.dot(np.linalg.inv(K_modif).dot(y)))

        return beta_hat

    def neg_marg_log_likelihood(self, y, X, theta, sigma_n, X_var=False, beta=0., h=lambda x:x, likelihood='gaussian'):
        """
        Returns:
            nmml: float
            -log p(y|X, theta, sigma, beta)
        """

        likelihood = self.process_type
        method = 'stable'
        
        # y - observable QoI values
        # H - matrix of vector functions -> after whitening could be identity functions?
        # beta - model coefficient vector (in basis functions) -> a function of theta
        # K - covariance matrix K(X,X|theta)
        # sigma_n - nugget value
        
        # theta - parameters of the kernel (sigma_f, l, [nu])
        # use logarithm with base 2
        # recalculate K based on X and theta 
        # mind fast matrix inversion or decomposition -> np.linalg.cholesky

        n = len(y)

        K = self.calc_covariance(X, self.kernel, theta['sigma_f'], theta['l'], n)
        #print('K={0}'.format(K)) ###DEBUG

        K_modif = K + (sigma_n**2) * np.eye(n)
        # TODO add option for heteroschedastic noise

        H = self.calc_H(self.X)

        # Use this to reduce optimisation space for GPR MLE fitting
        #beta_hat = self.beta_of_theta(y, sigma_n, theta['sigma_f'], theta['l'])
        #beta = beta_hat
        beta = np.zeros((n, self.n_y_dim)) # use assuming constant zero model y=0+f(x)

        M11 = y - np.dot(H, beta)

        if method == 'stable':
            
            #print('K_mod={0}'.format(K_modif)) ###DEBUG
            
            M2 = L = np.linalg.cholesky(K_modif)
            S1 = solve_triangular(L, M11, lower=True)
            M12 = S2 = solve_triangular(L.T, S1, lower=False)
            M2 = np.sum(np.log2(np.diag(L)))
      
        else:

            M12 = np.dot(np.linalg.inv(K_modif), M11)
            M2 = np.log2(np.linalg.det(K_modif))
        
        M1 = np.dot(M11.T, M12)

        # TODO: Think of better polymorphism with Python: 
        # - different function passed
        # - different class implemenetation for GPR/STP
        # - decorators? 
        
        # Option for GPR
        if likelihood == 'gaussian':
            
            mll = -0.5 * M1 \
              - 0.5 * n * np.log2(2 * np.pi) \
              - 0.5 * M2

            mll = mll[0][0]

        # Option for STP
        elif likelihood == 'student_t':

            nu = theta['nu'] 
            M1_m = ((nu - 2.) / nu) * M2
            mll = np.log2(gamma(0.5 * (nu + n))) \
                - np.log2(gamma(0.5 * nu)) \
                - 0.5 * n * np.log2(nu * np.pi) \
                - 0.5 * np.log2(M1_m) \
                - (0.5 * (nu + n)) * np.log2(1. + M1_m / nu)
                
                # K should include nugget and be K + sigma_n**2 * I_n ? -> now it includes
        
        # Use to find argmax of MLE over {beta, theta, sigma_n} 

        return -mll

    def nmll_hp(self, hpval):
        """
        Calculates marginal log likelohood for given data and kernel parameter value
        Function signature complies with scipy.optimize.minimize() 
        """  
        
        theta = {}

        theta['sigma_f'] = hpval[0]
        sigma_n = hpval[1]
        theta['nu'] = hpval[2]
        theta['l'] = hpval[3:] 

        self.sigma_f = theta['sigma_f'] 
        self.sigma_n = sigma_n
        self.nu = theta['nu']
        self.l = theta['l']

        self.fit_cov(self.X_train, self.X_train_var)

        return self.neg_marg_log_likelihood(self.y_train, self.X_train, theta, sigma_n, likelihood=self.process_type)

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

    r1 = np.divide(x - y, l) # TODO check that if l is array we get element-wise division
    r  = np.linalg.norm(r1)**2
    kernel = sigma_f * np.exp(-0.5 * r)

    #print('x={0};y={1};l={2};r1={3};r={4}'.format(x,y,l,r1,r)) ###DEBUG
    
    return kernel

def matern_kernel(x, y, sigma_f=1., l=1., nu=2.5):
    """
    Defines Mater kernel function
    """
    
    r1 = r1 = np.divide(x - y, l)
    r  = np.linalg.norm(r1)

    m1 = np.sqrt(2 * nu) * r

    if abs(nu - 0.5) < 1e-16:
        
        kernel = (sigma_f**2) * np.exp(-r)

    elif abs(nu - 1.5) < 1e-16:
        
        m2 = np.sqrt(3) * r
        kernel = (sigma_f**2) * (1 + m2) * np.exp(-m2)

    elif abs(nu - 2.5) < 1e-16:

        m2 = np.sqrt(5) * r
        kernel = (sigma_f**2) * (1 + m2 + (m2**2)/3.) * np.exp(-m2)
    
    else:
        
        kernel = (sigma_f**2) * (2**(1 - nu)) / gamma(nu) \
             * np.power(m1, nu) \
             * kv(m1, nu)
    
    return kernel
