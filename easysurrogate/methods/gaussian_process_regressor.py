import numpy as np
from itertools import product
from scipy.optimize import minimize


class GaussianProcess():

    def __init__(self):
        self.n_x_dim = 1

    def set_kernel(self, kernel='sq_exp'):
        if kernel == 'gibbs':
            self.kernel = gibbs_ns_kernel
        else:
            self.kernel = sq_exp_kernel_function

        self.sigma_f = 1.0
        self.sigma_n = 0.1
        self.l = 1.0

    def set_covariance(self, K):
        self.K = K
        self.n = K.shape[0]

    def fit_cov(self, X, covariance='regular'):

        K = [self.kernel(i, j, sigma_f=self.sigma_f, l=self.l) for (i, j) in product(X, X)]
        K = np.array(K).reshape(self.n, self.n)
        self.set_covariance(K)

        K_inv_tot = np.linalg.inv(self.K + (self.sigma_n ** 2) * np.eye(self.n))
        self.K_inv_tot = K_inv_tot

    def optmize_hyperparameters(self, X_train, y_train):
        """
        Optimizes hyperparamter values for minimum of R^2 score on training dataset
        #TODO has to optimise form MLE or MAP
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

        n_star = X_new.shape[0]
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
        K_star = np.array(K_star).reshape(n_star, self.n)

        f_bar_star = np.dot(K_star, np.dot(self.K_inv_tot, self.y.reshape(self.n, self.n_x_dim)))
        return f_bar_star

    def predict_var(self, X_new):

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
        K_star2 = np.array(K_star2).reshape(n_star, n_star)

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
        K_star = np.array(K_star).reshape(n_star, self.n)

        cov_f_star = K_star2 - np.dot(K_star, np.dot(self.K_inv_tot, K_star.T))
        var_f_star = np.diag(cov_f_star)

        return var_f_star

    def predict(self, X, return_std=True):

        y_mean = self.predict_mean(X)

        if return_std:
            y_var = self.predict_var(X)
            return y_mean, y_var

        return y_mean

    def r2_score(self, X, y):
        f = self.predict_mean(X)
        y_mean = y.mean()
        r2 = 1 - np.multiply(y - f, y - f).sum() / (np.multiply(y - y_mean, y - y_mean).sum())
        return r2

    def r2_score_hp(self, hpval):
        self.sigma_f = hpval[0]
        self.sigma_n = hpval[1]
        self.l = hpval[2]
        self.fit_cov(self.X_train)
        return self.r2_score(self.X_train, self.y_train)


def gibbs_ns_kernel(x, y, l, l_func=lambda x: x):
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
    """Define squared exponential kernel function."""
    kernel = sigma_f * np.exp(- (np.linalg.norm(x - y)**2) / (2 * l**2))
    return kernel
