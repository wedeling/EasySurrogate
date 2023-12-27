"""
Class for solving the forced-dissipative 2D vorticity equations.

Author: W. Edeling
"""

import numpy as np
from scipy import stats


class Vorticity_2D:
    """
    Class for solving the forced-dissipative 2D vorticity equations.
    """

    def __init__(self, N, dt, decay_time_nu, decay_time_mu, **kwargs):
        """
        Initialize the Vorticity_2D solver.

        Parameters
        ----------
        N : int
            The number of spatial points in 1 direction.
        dt : float
            The time step used in the 2nd order Adams Bashforth scheme.
        decay_time_nu : float
            The (e-folding) decay time of a Fourier mode at the finest scale
            in days when forced only by the viscous term. Used to set the
            viscosity coefficient nu.
        decay_time_mu : float
            The (e-folding) decay time of a Fourier mode at the finest scale
            in days when forced only by the forcing term. Used to set the
            forcing coefficient mu.
        kwargs
            Can be used to specify user-defined nu and mu values

        Returns
        -------
        None.

        """

        # number of points in a spatial direction
        self.N = N
        # time step
        self.dt = dt
        # cutoff in pseudospectral method
        self.Ncutoff = N / 3
        # compute computational grid
        self.x, self.y = self.get_grid()
        # get the spectral operators for the x, y gradients and Laplace operator
        self.kx, self.ky, self.k_squared, self.k_squared_no_zero = \
            self.get_derivative_operators()
        # get the spectral filter
        self.P = self._get_filter()

        # compute the viscosity and the forcing term coefficient
        Omega = 7.292 * 10**-5
        self.day = 24 * 60**2 * Omega
        nu = 1.0 / (self.day * self.Ncutoff**2 * decay_time_nu)
        mu = 1.0 / (self.day * decay_time_mu)

        # use values above unless user-defined nu, mu values are specified
        self.nu = kwargs.get('nu', nu)
        self.mu = kwargs.get('mu', mu)

        # forcing term
        F = 2**1.5 * np.cos(5 * self.x) * np.cos(5 * self.y)
        self.F_hat = np.fft.fft2(F)

        # constant factor that appears in AB/BDI2 time stepping scheme,
        # multiplying the Fourier coefficient w_hat_np1
        self.norm_factor = 1.0 / (3.0 / (2.0 * dt) - self.nu * self.k_squared + self.mu)

        self.binnumbers, self.bins = self.freq_map()
        self.N_bins = self.bins.size

    def get_grid(self):
        """
        Generate an equidistant N x N square grid in [0, 2*pi] x [0, 2*pi]

        Returns
        -------
        x, y: array
            The N x N coordinates of the grid.
        """

        # 2D grid
        N = self.N
        self.h = 2 * np.pi / N
        axis = self.h * np.arange(1, N + 1)
        axis = np.linspace(0, 2.0 * np.pi, N, endpoint=False)
        x, y = np.meshgrid(axis, axis)

        return x, y

    def get_derivative_operators(self):
        """
        Get the spectral operators for the gradients in x and y direction, plus
        the Laplace operators.

        Returns
        -------
        kx : array (complex)
            Operator for gradient in x direction.
        ky : array (complex)
            Operator for gradient in y direction.
        k_squared : array (complex)
            Laplace operator.
        k_squared_no_zero : array (complex)
            Laplace operator where the (0,0) entry is not zero. Used for computing
            the stream function.

        """

        N = self.N
        # 1D frequencies
        self.k_1d = np.fft.fftfreq(N) * N

        # kx = np.zeros([N, int(N / 2 + 1)]) + 0.0j
        # ky = np.zeros([N, int(N / 2 + 1)]) + 0.0j
        kx = np.zeros([N, N]) + 0.0j
        ky = np.zeros([N, N]) + 0.0j

        for i in range(N):
            # for j in range(int(N / 2 + 1)):
            for j in range(N):
                kx[i, j] = 1j * self.k_1d[j]
                ky[i, j] = 1j * self.k_1d[i]

        k_squared = kx**2 + ky**2
        k_squared_no_zero = np.copy(k_squared)
        k_squared_no_zero[0, 0] = 1.0

        return kx, ky, k_squared, k_squared_no_zero

    def _get_filter(self):
        '''
        Compute the spectral filter used to remove aliasing. Any wave number higher
        than the cutoff value N/3 will be set to zero.

        Returns
        -------
        None.

        '''
        N = self.N
        cutoff = self.Ncutoff
        # P = np.ones([N, int(N / 2 + 1)]) # for use in rfft
        P = np.ones([N, N])

        for i in range(N):
            # for j in range(int(N / 2 + 1)):  # for use in rfft
            for j in range(N):
                if np.abs(self.kx[i, j]) > cutoff or np.abs(self.ky[i, j]) > cutoff:
                    P[i, j] = 0.0

        return P

    def get_filter(self):
        return self.P

    def _get_Gaussian_filter(self):
        return np.exp(self.k_squared * self.h **2 / 24).real
    
    def get_scale_aware_filter(self, k_min, k_max):
        """
        Get a circular "scale-aware" spectral filter, that is only 1 for
        k_min <= k <= k_max. 

        Here k is the 1D wave number corresponding to all 2D (k_1, k_2) 
        wave numbers which fall in this 1D interval:

        k - 1/2 <= sqrt(k_1 **2 + k_2 ** 2) <= k + 1/2

        k = 0, 1, 2, ...

        To plot a scale-aware filter use e.g.

        P_k = scale_aware_filter(16, 21)
        vort_solver.plot_filter(P = P_k)

        Here vort_solver is a Vorticity_2D object.

        Parameters
        ----------
        k_min : int
            The minimum wave number.
        k_max : int
            The maximum wave number.

        Returns
        -------
        array, shape (N,N)
            The scale-aware spectral filter.

        """

        P_k = np.zeros([self.N, self.N])    
        idx0, idx1 = np.where((self.binnumbers >= k_min) & (self.binnumbers <= k_max))
        P_k[idx0, idx1] = 1.0

        return P_k[0:self.N, 0:self.N] 

    def plot_filter(self, **kwargs):
        """
        Plot a 2D image of the spectral filter P, vs the 2D wave number (k_1, k_2).
        A (k_1, k_2) point that is black corresponds to 1. white to 0.

        Parameters
        ----------
        **kwargs : array, shape (N, N)
            If P = some_other_filter is specified, this is plotted instead.
            If not, self.P is plotted.

        Returns
        -------
        None.

        """

        P = kwargs.get('P', self.P)

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel=r'$k_1$', ylabel=r'$k_2$')
        ax.imshow(P, cmap='gray_r')
        skip = 10
        ax.set_xticks(np.arange(self.N)[0:-1:skip])
        ax.set_xticklabels(self.k_1d[0:-1:skip].astype('int'))
        ax.set_yticks(np.arange(self.N)[0:-1:skip])
        ax.set_yticklabels(self.k_1d[0:-1:skip].astype('int'))
        plt.tight_layout()
        plt.show()

    def initial_cond(self):
        """
        Compute the initial condition.

        Returns
        -------
        w_hat_n : array (complex)
            The Fourier coefficient of the vorticity at t_{0}.
        w_hat_nm1 : array (complex)
            The Fourier coefficient of the vorticity at t_{-1}.
        VgradW_hat_nm1 : array (complex)
            The Fourier coefficients of the Jacobian (nonlinear advection term)
            at t_{-1}.

        """
        x = self.x
        y = self.y
        # initial condition
        w = np.sin(4.0 * x) * np.sin(4.0 * y) + 0.4 * np.cos(3.0 * x) * np.cos(3.0 * y) + \
            0.3 * np.cos(5.0 * x) * np.cos(5.0 * y) + 0.02 * np.sin(x) + 0.02 * np.cos(y)

        # initial Fourier coefficients at time n and n-1
        w_hat_n = self.P * np.fft.fft2(w)
        w_hat_nm1 = np.copy(w_hat_n)

        # initial Fourier coefficients of the jacobian at time n and n-1
        VgradW_hat_nm1 = self.compute_VgradW_hat(w_hat_n)

        return w_hat_n, w_hat_nm1, VgradW_hat_nm1

    def compute_VgradW_hat(self, w_hat_n):
        """
        Pseudo-spectral technique to solve for Fourier coefs of Jacobian.


        Parameters
        ----------
        w_hat_n : array (complex)
            The Fourier coefficient of the vorticity at t_{0}.

        Returns
        -------
        VgradW_hat_n : array (complex)
            The Fourier coefficients of the Jacobian (nonlinear advection term)
            at t_{n}.

        """
        # compute streamfunction
        psi_hat_n = self.compute_stream_function(w_hat_n)

        # compute jacobian in physical space
        u_n = np.fft.ifft2(-self.ky * psi_hat_n).real
        w_x_n = np.fft.ifft2(self.kx * w_hat_n).real

        v_n = np.fft.ifft2(self.kx * psi_hat_n).real
        w_y_n = np.fft.ifft2(self.ky * w_hat_n).real

        VgradW_n = u_n * w_x_n + v_n * w_y_n

        # return to spectral space
        VgradW_hat_n = np.fft.fft2(VgradW_n)

        VgradW_hat_n *= self.P

        return VgradW_hat_n

    def compute_stream_function(self, w_hat):
        """
        Compte the stream function

        Parameters
        ----------
        w_hat : array (complex)
            The Fourier coefficients of the vorticity.

        Returns
        -------
        psi_hat : array (complex)
            The Fourier coefficients of the stream function.

        """
        psi_hat = w_hat / self.k_squared_no_zero
        psi_hat[0, 0] = 0.0
        return psi_hat

    def step(self, w_hat_n, w_hat_nm1, VgradW_hat_nm1, sgs_hat=0.0):
        """
        Solve for the vorticity at time n+1 and the Jacobian at time n.

        Parameters
        ----------
        w_hat_n : array (complex)
            The Fourier coefficient of the vorticity at t_{n}.
        w_hat_nm1 : array (complex)
            The Fourier coefficient of the vorticity at t_{n-1}.
        VgradW_hat_nm1 : array (complex)
            The Fourier coefficients of the Jacobian (nonlinear advection term)
            at t_{n-1}.
        sgs_hat : array (complex)
            The Fourier coefficients of the sungrid-scale term. The default is 0.0.

        Returns
        -------
        w_hat_np1 : array (complex)
            The Fourier coefficient of the vorticity at t_{n+1}.
        VgradW_hat_n : array (complex)
            The Fourier coefficients of the Jacobian (nonlinear advection term)
            at t_{n}.

        """
        # compute the Jacobian
        VgradW_hat_n = self.compute_VgradW_hat(w_hat_n)
        # solve for next time step according to AB/BDI2 scheme
        w_hat_np1 = self.norm_factor * self.P * (2.0 / self.dt * w_hat_n -
                                                 1.0 / (2.0 * self.dt) * w_hat_nm1 -
                                                 2.0 * VgradW_hat_n + VgradW_hat_nm1 +
                                                 self.mu * self.F_hat - sgs_hat)
        return w_hat_np1, VgradW_hat_n

    def down_scale(self, X_hat, N_LR):
        """
        Down-scale X to a lower spatial resolution by removing high-frequency Fourier coefficients.

        Parameters
        ----------
        X_hat : array (complex)
            An array of Fourier coefficients of X, where the spatial resolution in 1D is higher than N.
        N_LR : int
            The new, lower, spatial resolution (X.shape[0] < N).

        Returns
        -------
        X_hat : array (complex)
            The Fourier coefficients of X at a spatial resolution determined by N.

        """
        assert N_LR < X_hat.shape[0], "N must be smaller than X_hat.shape[0] to down scale."

        # The spatial dimension of the problem
        d = X_hat.ndim
        # The HR grid resolution
        N_HR = X_hat.shape[0]

        # the range that should be deleted
        start = int(N_LR / 2)
        end = X_hat.shape[0] - start
        # Remove the Fourier coefficients that are not present in the lower-resolution version
        # of X_hat. Index 'start' needs +1 in case N_LR is odd.
        for i in range(X_hat.ndim):
            X_hat = np.delete(X_hat, np.arange(start + np.mod(N_LR, 2), end), axis=i)
        # In numpy only the inverse transform is scaled. The following term must be applied
        # to ensure the correct scaling factor is applied in the inverse transform.
        scaling_factor = (N_LR / N_HR)**d
        return X_hat * scaling_factor

    def freq_map(self):
        """
        Map 2D frequencies to a 1D bin (kx, ky) --> k,
        where k = 0, 1, ..., sqrt(2)*Ncutoff.

        Returns
        -------
        binnumbers : array
            A 2D array containing the distance k of each freq pair (kx, ky).
        bins : array
            The edges of the 1D freq bins.

        """

        # edges of 1D wavenumber bins
        bins = np.arange(-0.5, np.ceil(2**0.5 * self.Ncutoff) + 1)
        #fmap = np.zeros([N,N]).astype('int')

        N = self.N
        dist = np.zeros([N, N])

        for i in range(N):
            for j in range(N):
                # Euclidian distance of frequencies kx and ky
                dist[i, j] = np.sqrt(self.kx[i, j]**2 + self.ky[i, j]**2).imag

        # find 1D bin index of dist
        _, _, binnumbers = stats.binned_statistic(dist.flatten(), np.zeros(self.N**2), bins=bins)
        binnumbers -= 1

        return binnumbers.reshape([N, N]), bins

    def EZ_spectrum(self, w_hat):
        """
        Compute the spectrum of the energy and enstrophy

        Parameters
        ----------
        w_hat : array (complex)
            The Fourier coefficients of the vorticity.

        Returns
        -------
        E_spec : array
            The energy per frequency bin k = 0, 1, ..., sqrt(2)*Ncutoff.
            These bins are computed in the freq_map subroutine.
        Z_spec : array
            The enstrophy per frequency bin k = 0, 1, ..., sqrt(2)*Ncutoff.

        """

        N = self.N

        psi_hat = w_hat / self.k_squared_no_zero
        psi_hat[0, 0] = 0.0

        E_hat = -0.5 * psi_hat * np.conjugate(w_hat) / N**4
        Z_hat = 0.5 * w_hat * np.conjugate(w_hat) / N**4

        E_spec = np.zeros(self.N_bins)
        Z_spec = np.zeros(self.N_bins)

        for i in range(N):
            for j in range(N):
                bin_idx = self.binnumbers[i, j]
                E_spec[bin_idx] += E_hat[i, j].real
                Z_spec[bin_idx] += Z_hat[i, j].real

        return E_spec, Z_spec

    def spectrum(self, a_hat):
        """
        Compute the spectrum of the 2D Fourier coefficients a_hat.

        Parameters
        ----------
        a_hat : array, complex
            The Fourier coefficients.

        Returns
        -------
        a_spec : array
            The spectrum of a_hat per frequency bin k = 0, 1, ..., sqrt(2)*Ncutoff.
            These bins are computed in the freq_map subroutine.

        """

        N = self.N

        a_spec = np.zeros(self.N_bins)

        for i in range(N):
            for j in range(N):
                bin_idx = self.binnumbers[i, j]
                a_spec[bin_idx] += a_hat[i, j].real

        return a_spec
