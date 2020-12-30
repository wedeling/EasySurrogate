"""
A 2D Gray-Scott reaction diffusion model with periodic boundary conditions.

Numerical method:
Craster & Sassi, spectral algorithmns for reaction-diffusion equations, 2006.

Code: W. Edeling

"""

import pickle
import numpy as np


class GrayScott_2D:
    """
    GrayScott_2D class
    """

    def __init__(self, N, L, dt, feed, kill, epsilon_u, epsilon_v):
        """
        Initialize a GrayScott_2D object

        Parameters
        ----------
        N : int
            The number of spatial points in one direction.
        L : float
            The size / scaling of the spatial domain in one direction.
        dt : float
            Time step.
        feed : float.
            Model parameter.
        kill : float
            Model paramater.
        epsilon_u : float
            Model paramater.
        epsilon_v : float
            Model parameter.

        Returns
        -------
        None.

        """
        self.N = N
        self.L = L
        self.dt = dt
        self.feed = feed
        self.kill = kill
        self.epsilon_u = epsilon_u
        self.epsilon_v = epsilon_v

        # 2D grid, scaled by L
        self.xx, self.yy = self.get_grid()

        # spatial derivative operators
        kx, ky = self.get_derivative_operator()

        # Laplace operator
        self.k_squared = kx**2 + ky**2

        # Integrating factors
        int_fac_u, int_fac_u2, int_fac_v, int_fac_v2 = self.integrating_factors()
        self.int_fac_u = int_fac_u
        self.int_fac_u2 = int_fac_u2
        self.int_fac_v = int_fac_v
        self.int_fac_v2 = int_fac_v2

    def get_grid(self):
        """
        Generate an equidistant N x N square grid

        Returns
        -------
        xx, yy: array
            the N x N coordinates

        """
        x = (2 * self.L / self.N) * np.arange(-self.N / 2, self.N / 2)
        y = x
        xx, yy = np.meshgrid(x, y)
        return xx, yy

    def get_derivative_operator(self):
        """
        Get the spectral operators used to compute the spatial dervatives in
        x and y direction

        Parameters
        ----------
        N : int
            number of points in 1 dimension

        Returns
        -------
        kx, ky: array (complex)
            operators to compute derivatives in spectral space. Already multiplied by
            the imaginary unit 1j

        """
        N = self.N
        # frequencies of fft2
        k = np.fft.fftfreq(N) * N
        # frequencies must be scaled as well
        k = k * np.pi / self.L
        kx = np.zeros([N, N]) + 0.0j
        ky = np.zeros([N, N]) + 0.0j

        for i in range(N):
            for j in range(N):
                kx[i, j] = 1j * k[j]
                ky[i, j] = 1j * k[i]

        return kx, ky

    def initial_cond(self):
        """
        Compute the initial condition

        Returns
        -------
        u_hat, v_hat: array(complex)
            initial Fourier coefficients of u and v

        """
        common_exp = np.exp(-10 * (self.xx**2 / 2 + self.yy**2)) + \
            np.exp(-50 * ((self.xx - 0.5)**2 + (self.yy - 0.5)**2))
        u = 1 - 0.5 * common_exp
        v = 0.25 * common_exp
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)

        return u_hat, v_hat

    def integrating_factors(self):
        """
        Compute the integrating factors used in the RK4 time stepping

        Parameters
        ----------
        k_squared : array(complex)
            the operator to compute the Laplace operator

        Returns
        -------
        The integrating factors for u and v

        """

        int_fac_u = np.exp(self.epsilon_u * self.k_squared * self.dt / 2)
        int_fac_u2 = np.exp(self.epsilon_u * self.k_squared * self.dt)
        int_fac_v = np.exp(self.epsilon_v * self.k_squared * self.dt / 2)
        int_fac_v2 = np.exp(self.epsilon_v * self.k_squared * self.dt)

        return int_fac_u, int_fac_u2, int_fac_v, int_fac_v2

    def rhs_hat(self, u_hat, v_hat, **kwargs):
        """
        Right hand side of the 2D Gray-Scott equations

        Parameters
        ----------
        u_hat : Fourier coefficients of u
        v_hat : Fourier coefficients of v

        Returns
        -------
        The Fourier coefficients of the right-hand side of u and v (f_hat & g_hat)

        """

        if 'reduced_sgs_u' in kwargs and 'reduced_sgs_v' in kwargs:
            reduced_sgs_u = kwargs['reduced_sgs_u']
            reduced_sgs_v = kwargs['reduced_sgs_v']
        else:
            reduced_sgs_u = reduced_sgs_v = 0

        if 'nudge_u_hat' in kwargs and 'nudge_v_hat' in kwargs:
            nudge_u_hat = kwargs['nudge_u_hat']
            nudge_v_hat = kwargs['nudge_v_hat']
            nudge_u = np.fft.ifft2(nudge_u_hat)
            nudge_v = np.fft.ifft2(nudge_v_hat)
        else:
            nudge_u = nudge_v = 0.0

        u = np.fft.ifft2(u_hat)
        v = np.fft.ifft2(v_hat)

        f = -u * v * v + self.feed * (1 - u) - reduced_sgs_u + nudge_u
        g = u * v * v - (self.feed + self.kill) * v - reduced_sgs_v + nudge_v

        f_hat = np.fft.fft2(f)
        g_hat = np.fft.fft2(g)

        return f_hat, g_hat

    def rk4(self, u_hat, v_hat, **kwargs):
        """
        Runge-Kutta 4 time-stepping subroutine

        Parameters
        ----------
        u_hat : Fourier coefficients of u
        v_hat : Fourier coefficients of v

        Returns
        -------
        u_hat and v_hat at the next time step

        """
        # RK4 step 1
        k_hat_1, l_hat_1 = self.rhs_hat(u_hat, v_hat, **kwargs)
        k_hat_1 *= self.dt
        l_hat_1 *= self.dt
        u_hat_2 = (u_hat + k_hat_1 / 2) * self.int_fac_u
        v_hat_2 = (v_hat + l_hat_1 / 2) * self.int_fac_v
        # RK4 step 2
        k_hat_2, l_hat_2 = self.rhs_hat(u_hat_2, v_hat_2, **kwargs)
        k_hat_2 *= self.dt
        l_hat_2 *= self.dt
        u_hat_3 = u_hat * self.int_fac_u + k_hat_2 / 2
        v_hat_3 = v_hat * self.int_fac_v + l_hat_2 / 2
        # RK4 step 3
        k_hat_3, l_hat_3 = self.rhs_hat(u_hat_3, v_hat_3, **kwargs)
        k_hat_3 *= self.dt
        l_hat_3 *= self.dt
        u_hat_4 = u_hat * self.int_fac_u2 + k_hat_3 * self.int_fac_u
        v_hat_4 = v_hat * self.int_fac_v2 + l_hat_3 * self.int_fac_v
        # RK4 step 4
        k_hat_4, l_hat_4 = self.rhs_hat(u_hat_4, v_hat_4, **kwargs)
        k_hat_4 *= self.dt
        l_hat_4 *= self.dt
        u_hat = u_hat * self.int_fac_u2 + 1 / 6 * (k_hat_1 * self.int_fac_u2 +
                                                   2 * k_hat_2 * self.int_fac_u +
                                                   2 * k_hat_3 * self.int_fac_u +
                                                   k_hat_4)
        v_hat = v_hat * self.int_fac_v2 + 1 / 6 * (l_hat_1 * self.int_fac_v2 +
                                                   2 * l_hat_2 * self.int_fac_v +
                                                   2 * l_hat_3 * self.int_fac_v +
                                                   l_hat_4)
        self.u_hat = u_hat
        self.v_hat = v_hat

        return u_hat, v_hat

    def store_state(self, fname):
        """
        Store the state to a pickle file.

        Parameters
        ----------
        fname : string
            The filename to which the state must be stored.

        Returns
        -------
        None.

        """

        f = open(fname, 'wb')
        pickle.dump({'u_hat': self.u_hat, 'v_hat': self.v_hat}, f)
        f.close()
        print('Saved state to %s.' % f.name)

    def load_state(self, fname):
        """
        Load the state from file.

        Parameters
        ----------
        fname : string
            The filename containing the state.

        Returns
        -------
        u_hat : array (complex)
            The Fourier coefficient of u.
        v_hat : array (complex)
            The Fourier coefficient of v.

        """

        f = open(fname, 'rb')
        state = pickle.load(f)
        return state['u_hat'], state['v_hat']
