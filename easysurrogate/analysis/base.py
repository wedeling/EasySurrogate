"""
The Base analysis class which all other analysis classes inherit. Contains
general functions which can be of use for various analysis objects, e.g.
computing a kernel density estimate.
"""

import numpy as np
from sklearn.neighbors import KernelDensity


class BaseAnalysis:
    """
    A Basic analysis class
    """

    def __init__(self):
        """
        Constructor
        """
        self.name = 'Base_Analysis'

    def auto_correlation_function(self, X, max_lag):
        """
        Compute the autocorrelation of X over max_lag time steps

        Parameters:
            - X (array, size (N,)): the samples from which to compute the ACF
            - max_lag (int): the max number of time steps, determines max
              lead time

        Returns:
            - R (array): array of ACF values
        """
        lags = np.arange(1, max_lag)
        R = np.zeros(lags.size)

        idx = 0

        print('Computing auto-correlation function')

        # for every lag, compute autocorrelation:
        # R = E[(X_t - mu_t)*(X_s - mu_s)]/(std_t*std_s)
        for lag in lags:

            X_t = X[0:-lag]
            X_s = X[lag:]

            mu_t = np.mean(X_t)
            std_t = np.std(X_t)
            mu_s = np.mean(X_s)
            std_s = np.std(X_s)

            R[idx] = np.mean((X_t - mu_t) * (X_s - mu_s)) / (std_t * std_s)
            idx += 1

        print('done')

        e_fold_idx = np.where(R <= np.e**-1)[0][0]
        print('E-folding index = %d' % e_fold_idx)

        return R

    def cross_correlation_function(self, X, Y, max_lag):
        """
        Compute the crosscorrelation between X and Y over max_lag time steps

        Parameters:
            - X, Y (array, size (N,)): the samples from which to compute the CCF
            - max_lag (int): the max number of time steps, determines max
              lead time

        Returns:
            - C (array): array of CCF values
        """
        lags = np.arange(1, max_lag)
        C = np.zeros(lags.size)

        idx = 0

        print('Computing cross-correlation function')

        # for every lag, compute cross correlation:
        # R = E[(X_t - mu_Xt)*(Y_s - mu_Ys)]/(std_Xt*std_Ys)
        for lag in lags:

            X_t = X[0:-lag]
            Y_s = Y[lag:]

            mu_t = np.mean(X_t)
            std_t = np.std(X_t)
            mu_s = np.mean(Y_s)
            std_s = np.std(Y_s)

            C[idx] = np.mean((X_t - mu_t) * (Y_s - mu_s)) / (std_t * std_s)
            idx += 1

        print('done')

        return C

    def get_pdf(self, X, Npoints=100):
        """
        Computes a kernel density estimate of the samples in X

        Parameters:
            - X (array): the samples
            - Npoints (int, default = 100): the number of points in the domain of X

        Returns:
            - domain (array of size (Npoints,): the domain of X
            - kde (array of size (Npoints,): the kernel-density estimate
        """

        #    kernel = stats.gaussian_kde(X, bw_method='scott')
        #    x = np.linspace(np.min(X), np.max(X), Npoints)
        #    pde = kernel.evaluate(x)
        #    return x, pde

        print('Computing kernel-density estimate')

        X_min = np.min(X)
        X_max = np.max(X)
        bandwidth = (X_max - X_min) / 40

        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X.reshape(-1, 1))
        domain = np.linspace(X_min, X_max, Npoints).reshape(-1, 1)
        log_dens = kde.score_samples(domain)

        print('done')

        return domain, np.exp(log_dens)

    def get_confidence_intervals(self, samples, conf=0.9):
        """
        Compute the confidence intervals given an array of samples

        Parameters
        ----------
        samples : array
            Samples on which to compute the intervals.
        conf : float, optional, must be in [0, 1].
            The confidence interval percentage. The default is 0.9.

        Returns
        -------
        lower : array
            The lower confidence bound..
        upper : array
            The upper confidence bound.

        """

        # ake sure conf is in [0, 1]
        if conf < 0.0 or conf > 1.0:
            print('conf must be specified within [0, 1]')
            return

        # lower bound = alpha, upper bound = 1 - alpha
        alpha = 0.5 * (1.0 - conf)

        # arrays for lower and upper bound of the interval
        n_samples = samples.shape[0]
        N_qoi = samples.shape[1]
        lower = np.zeros(N_qoi)
        upper = np.zeros(N_qoi)

        # the probabilities of the ecdf
        prob = np.linspace(0, 1, n_samples)
        # the closest locations in prob that correspond to the interval bounds
        idx0 = np.where(prob <= alpha)[0][-1]
        idx1 = np.where(prob <= 1.0 - alpha)[0][-1]

        # for every location of qoi compute the ecdf-based confidence interval
        for i in range(N_qoi):
            # the sorted surrogate samples at the current location
            samples_sorted = np.sort(samples[:, i])
            # the corresponding confidence interval
            lower[i] = samples_sorted[idx0]
            upper[i] = samples_sorted[idx1]

        return lower, upper

    def recursive_moments(self, X_np1, mu_n, sigma2_n, N):
        """
        Update the mean and variance using a new data sample X_np1.

        Parameters
        ----------
        X_np1 : aray
            A new data point, iteration n+1
        mu_n : array
            The mean compute using all samples up to iteration n
        sigma2_n : array
            The variance compute using all samples up to iteration n.
        N : int
            The number of samples.

        Returns
        -------
        mu_np1 : array
            The new mean, updated with the latest data point.
        sigma2_np1 : TYPE
            The new variance, updated with the latest data point..

        """

        mu_np1 = mu_n + (X_np1 - mu_n) / (N + 1)

        sigma2_np1 = sigma2_n + mu_n**2 - mu_np1**2 + (X_np1**2 - sigma2_n - mu_n**2) / (N + 1)

        return mu_np1, sigma2_np1
