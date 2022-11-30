"""
Class and functions for Gaussian Process Regression model
------------------------------------------------------------------------------
Author: Y. Yudin
==============================================================================
"""

import numpy as np
import functools
import math

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

        if 'nu_stp' not in kwargs:
            self.nu_stp = 3
        else:
            self.nu_stp = kwargs['nu_stp']

        if 'nu_matern' not in kwargs:
            self.nu_matern = 2.5
        else:
            self.nu_matern = kwargs['nu_matern']

        # Set the type of the stochastic process
        if 'process_type' not in kwargs:
            self.process_type = 'gaussian'
        else:
            self.process_type = kwargs['process_type']

        # Set the kernel type
        if 'kernel' not in kwargs:
            self.kernel_name = 'sq_exp'
        else:
            self.kernel_name = kwargs['kernel']
        self.set_kernel(self.kernel_name)

    def __del__(self):

        print('>Destructor[Surrogate]: called')
        print('>Destructor[Surrogate]: The kernel function was called times: {0}'.
            format(self.kernel.func.call_counter if type(self.kernel)==functools.partial else self.kernel.call_counter)) ###DEBUG

    def set_kernel(self, kernel='sq_exp', **kwargs):
        """
        Setting type of the kernel function and its default parameters
        """

        if kernel == 'gibbs':
            self.kernel = gibbs_ns_kernel
        elif kernel == 'matern':
            #self.kernel = matern_kernel
            #self.kernel = lambda x: matern_kernel(x, nu=self.nu_matern)
            self.kernel = functools.partial(matern_kernel, nu=self.nu_matern)
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
        
        #print('K[0][0]={0}, X[0]={1}, sigma_n={4}, sigma_f={2}, l={3}, nu_stp={5}'.format(K[0], X[0], sigma_f, l, self.sigma_n, self.nu_stp)) ###DEBUG
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
            K_modif = self.K + (self.sigma_n ** 2) * np.eye(self.n)
        else:
            Eta = self.calc_noise(X_var)
            K_modif = self.K + Eta
                    
        K_inv_modif = np.linalg.inv(K_modif)
        
        self.K_modif = K_modif
        self.K_inv_modif = K_inv_modif

    def optmize_hyperparameters(self, X_train, y_train, X_train_var=False, loss='r2'):
        """
        Optimizes hyperparamter values for minimum of R^2 score on training dataset OR perform MLE
        #TODO has to optimise from MLE or MAP -> MLE done
        Uses scipy .minimize() to find the optimum
        Reassigns the attributes of the object after optimization; assumes the covariance matrix is calculated before the first iteration
        """

        self.X_train = X_train
        self.y_train = y_train
        self.X_train_var = X_train_var

        hp_curval = np.array([self.sigma_f, self.sigma_n, self.nu_stp, *self.l,])

        # Mind the bounds, sometimes the lenghtscale is too small -> it is a tradeoff between optimizer and further usage
        #hp_bounds = [(1e-16, 1e+6), (1e-16, 1e+4), (2, 1e+6), *([(1e-16, 1e+4)]*len(self.l))]
        hp_bounds = [(1e-6, 1e+6), (1e-6, 1e+4), (2, 1e+6), *([(1e-6, 1e+4)]*len(self.l))]

        hp_options = {'maxiter': 100}

        if loss == 'nmll':
            loss_func = self.nmll_hp
        else:
            loss_func == self.r2_score_hp

        hp_optval_res = minimize(loss_func, hp_curval, bounds=hp_bounds, options=hp_options)

        hp_optval = hp_optval_res.x

        print('Optimisation results: [{0}] with the optimum at {1}'.format(hp_optval_res.message, hp_optval_res.x))

        self.sigma_f = hp_optval[0]
        self.sigma_n = hp_optval[1]     
        #self.nu_stp = hp_optval[2] # for now setting nu as fixed during optimisation
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

        #self.fit_cov(X, X_var)

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

        f_bar_star = np.dot(K_star, np.dot(self.K_inv_modif, self.y.reshape(self.n, self.n_y_dim)))

        print('resulting_mean={0}, sigma_f={1}, l={2} \n'.format(f_bar_star,self.sigma_f, self.l,)) ###DEBUG
        #print('K*={0}\n'.format(K_star)) ###DEBUG
        
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

        M1 = K_star2 - np.dot(K_star, np.dot(self.K_inv_modif, K_star.T))

        if likelihood == 'gaussian':
            
            cov_f_star = M1

        elif likelihood == 'student_t':
            
            M2 = np.dot(self.y.T, np.dot(self.K_inv_modif, self.y))
            cov_f_star = np.dot((self.nu_stp + M2 - 2)/(self.nu_stp + self.n - 2), M1)

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
        self.nu_stp = hpval[2]
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

    def beta_of_theta(self, y, sigma_n, sigma_f, l, X_var=False):
        """
        Estimate optimal beta ( coefficient vector for base model of y = h(x)*beta+f(x) ) for given parameters of kernel theta: (sigma_f, l) 
        """

        n = len(y)

        K = self.calc_covariance(self.X, self.kernel, sigma_f, l, n)
        
        if not X_var:
            K_modif = self.K + (self.sigma_n ** 2) * np.eye(self.n)
        else:
            Eta = self.calc_noise(X_var)
            K_modif = self.K + Eta

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

        K = self.calc_covariance(X, self.kernel, sigma_f=theta['sigma_f'], l=theta['l'], n=n)
        
        #print('K={0}'.format(K)) ###DEBUG
        
        #print('K[0][0]={0}, X[0]={1}, sigma_n={4}, sigma_f={2}, l={3}, nu_stp={5}'.\
        #    format(K[0][0], X[0], theta['sigma_f'], theta['l'], sigma_n, self.nu_stp)) ###DEBUG

        if not X_var:
            K_modif = K + (sigma_n ** 2) * np.eye(n)
        else:
            Eta = self.calc_noise(X_var)
            K_modif = K + Eta
        # TODO add option for heteroschedastic noise
        #print('K_modif={0}'.format(K_modif)) ###DEBUG

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
        M1 = M1[0][0]

        # TODO: Think of better polymorphism with Python: 
        #  - different function passed
        #  - different class implemenetation for GPR/STP
        #  - decorators? 
        # + if no implementation -> throw exceptions/log errors
        
        # Option for GPR
        if likelihood == 'gaussian':
            
            mll = -0.5 * M1 \
              - 0.5 * n * np.log2(2 * np.pi) \
              - 0.5 * M2

        # Option for STP
        elif likelihood == 'student_t':

            nu = theta['nu'] 
            alpha = ((nu - 2.) / nu)
            beta = 0.5 * (nu + n)

            mll = np.log2(gamma(beta)) \
                - np.log2(gamma(0.5 * nu)) \
                - 0.5 * n * np.log2(nu * np.pi) \
                - 0.5 * np.log2(alpha) \
                - 0.5 * M2 \
                - beta * np.log2(1. + alpha * M1 / nu)
                
                # K should include nugget and be K + sigma_n**2 * I_n ? -> now it includes
                # should the 4th term be there?
        
        # Use to find argmax of MLE over {beta, theta, sigma_n} 

        #print('nmll={0}'.format(-mll)) ###DEBUG
        return -mll

    def nmll_hp(self, hpval):
        """
        Calculates marginal log likelohood for given data and kernel parameter value
        Function signature complies with scipy.optimize.minimize() 
        """  
        
        theta = {}

        theta['sigma_f'] = hpval[0]
        
        sigma_n = hpval[1]
        
        #theta['nu'] = hpval[2] # fixing nu during optimisation
        theta['nu'] = self.nu_stp

        theta['l'] = hpval[3:] 

        """
        self.sigma_f = theta['sigma_f'] 
        self.sigma_n = sigma_n
        self.nu_stp = theta['nu']
        self.l = theta['l']
        """
        #self.fit_cov(self.X_train, self.X_train_var)

        return self.neg_marg_log_likelihood(self.y_train, self.X_train, theta, sigma_n, likelihood=self.process_type)

########################################
### Kernel functions implementations ###
########################################

def set_nu(func, nu):
    def wrapper(*args, **kwargs):
        kwargs['nu'] = nu
        val = func(*args, **kwargs)  
        return val
    return wrapper

def counted(func):
    func.call_counter = 0
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        wrapped.call_counter += 1
        return func(*args, **kwargs)
    return wrapped

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
    Defines squared exponential kernel function, same as RBF
    """

    r1 = np.divide(x - y, l)
    r  = np.linalg.norm(r1)**2
    kernel = sigma_f * np.exp(-0.5 * r)

    #print('x={0};y={1};l={2};r1={3};r={4}'.format(x,y,l,r1,r)) ###DEBUG
    
    return kernel

def exp_kernel_function(x, y, sigma_f=1., l=1.):
    """
    Defines exponential kernel function
    """

    r1 = np.divide(x - y, l)
    r  = np.linalg.norm(r1)**2
    kernel = sigma_f * np.exp(-1. * np.sqrt(r))
    
    return kernel

@counted
def matern_kernel(x, y, sigma_f=1., l=1., nu=2.5):
    """
    Defines Matern kernel function
    """
    
    r1 = np.divide(x - y, l)
    r  = np.linalg.norm(r1)

    m1 = np.sqrt(2 * nu) * r

    if math.isclose(nu, 0.5):
        
        kernel = (sigma_f**2) * np.exp(-r)

    elif math.isclose(nu, 1.5):
        
        m2 = np.sqrt(3) * r
        kernel = (sigma_f**2) * (1 + m2) * np.exp(-m2)
        #NB: np.exp(-1E5)==0.0
        #TODO: some passed l is 1E-16

    elif math.isclose(nu, 2.5):

        m2 = np.sqrt(5) * r
        kernel = (sigma_f**2) * (1 + m2 + (m2**2)/3.) * np.exp(-m2)
    
    else:
        
        kernel = (sigma_f**2) * (2**(1 - nu)) / gamma(nu) \
             * np.power(m1, nu) \
             * kv(m1, nu)
    
    #if math.isclose(kernel, 0.):
    #    print('Matern-{0:.1f}({1},{2})={3}\n'.format(nu, x, y, kernel)) ###DEBUG

    #if min(abs(l)) < 1e-9:
    #    print('Lengthscale is {0} \n'.format(l)) ###DEBUG

    return kernel
