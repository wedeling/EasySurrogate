"""
Lorenz 96 solver (2 Layers)
"""
import numpy as np
import easysurrogate as es
import matplotlib.pyplot as plt


class L96:
    """
    Class for the 2-layer Lorenz 96 model.
    """

    def __init__(self, dt, K=18, J=20, F=10, h_x=-1, h_y=1, epsilon=0.5):
        """
        Initialize the L96 solver.

        Parameters
        ----------
        dt : float
            The time step.
        K : int, optional
            The number of macroscopic ODEs. The default is 18.
        J : int, optional
            The number of microscopic ODEs per macrscopic ODE. The default is 20.
        F : int, optional
            The constant forcing term. The default is 10.
        h_x : float, optional
            Model parameter. Regulates the influence of the microscopic scales on the
            macroscopic scales. The default is -1.
        h_y : float, optional
            Model parameter. Regulates the influence of the macroscopic scales on the
            microscopic scales. The default is 1.
        epsilon : float, optional
            Model parameter. Regulates the time-scale separation between macroscopic
            and microscopic scales. The default is 0.5.

        Returns
        -------
        None.

        """
        self.dt = dt
        self.K = K
        self.J = J
        self.F = F
        self.h_x = h_x
        self.h_y = h_y
        self.epsilon = epsilon

    def initial_conditions(self):
        """
        Generate the initial conditions.

        Returns
        -------
        X_n : array
            A vector of K macroscopic variables X.
        f_nm1 : array
            The right-hand sides of the X ODEs.

        """
        # equilibrium initial condition for X, zero IC for Y
        X_n = np.ones(self.K) * self.F
        X_n[10] += 0.01  # add small perturbation to 10th variable

        # initial condition small-scale variables
        self.Y_n = np.zeros([self.J, self.K])
        B_n = self.h_x * np.mean(self.Y_n, axis=0)

        # initial right-hand sides
        f_nm1 = self.rhs_X(X_n, B_n)

        # copmute the right-hand side of the Y ODEs.
        self.g_nm1 = np.zeros([self.J, self.K])
        for k in range(self.K):
            self.g_nm1[:, k] = self.rhs_Y_k(self.Y_n, X_n[k], k)

        return X_n, f_nm1

    def rhs_X(self, X, r):
        """
        Compute the right-hand side of the X ODEs

        Parameters
        ----------
        X : array
            The array containing the K macroscopic variables X_k.
        r : array
            The subgrid scale array.

        Returns
        -------
        rhs_X : array
            The array containing the right-hand sides of the X ODEs.

        """

        rhs_X = np.zeros(self.K)

        # first treat boundary cases (k=1, k=2 and k=K)
        rhs_X[0] = -X[self.K - 2] * X[self.K - 1] + X[self.K - 1] * X[1] - X[0] + self.F

        rhs_X[1] = -X[self.K - 1] * X[0] + X[0] * X[2] - X[1] + self.F

        rhs_X[self.K - 1] = -X[self.K - 3] * X[self.K - 2] + \
            X[self.K - 2] * X[0] - X[self.K - 1] + self.F

        # treat interior points
        for k in range(2, self.K - 1):
            rhs_X[k] = -X[k - 2] * X[k - 1] + X[k - 1] * X[k + 1] - X[k] + self.F

        rhs_X += r

        return rhs_X

    def rhs_Y_k(self, Y, X_k, k):
        """
        Compute the right-hand side of Y for fixed k

        Parameters
        ----------
        Y : array (size (J,K))
            Microscopic variables.
        X_k : float
            Macroscopic variable X_k
        k : int
            The index k.

        Returns
        -------
        rhs_Yk : array (size (J))
            The right-hand side of Y for fixed k

        """

        rhs_Yk = np.zeros(self.J)

        # first treat boundary cases (j=1, j=J-1, j=J-2)
        if k > 0:
            idx = k - 1
        else:
            idx = self.K - 1
        rhs_Yk[0] = (Y[1, k] * (Y[self.J - 1, idx] - Y[2, k]) -
                     Y[0, k] + self.h_y * X_k) / self.epsilon

        if k < self.K - 1:
            idx = k + 1
        else:
            idx = 0
        rhs_Yk[self.J - 2] = (Y[self.J - 1, k] *
                              (Y[self.J - 3, k] - Y[0, idx])
                              - Y[self.J - 2, k] + self.h_y * X_k) / self.epsilon
        rhs_Yk[self.J - 1] = (Y[0, idx] * (Y[self.J - 2, k] - Y[1, idx]) -
                              Y[self.J - 1, k] + self.h_y * X_k) / self.epsilon

        # treat interior points
        for j in range(1, self.J - 2):
            rhs_Yk[j] = (Y[j + 1, k] * (Y[j - 1, k] - Y[j + 2, k]) -
                         Y[j, k] + self.h_y * X_k) / self.epsilon

        return rhs_Yk

    def _step_X(self, X_n, f_nm1, r_n):
        """
        Integrate the X equation in time using Adams-Bashforth

        Parameters
        ----------
        X_n : array (size (K))
            The large scale variables at time n.
        f_nm1 : array (size (K))
            The right-hand side of X at time n-1.
        r_n : array (size (K))
            The subgrid scale term

        Returns
        -------
        X_np1 : array (size (K))
            The large scale variables at time n+1.
        f_n : array (size (K))
            The right-hand side of X at time n+1.

        """

        # right-hand side at time n
        f_n = self.rhs_X(X_n, r_n)

        # adams bashforth
        X_np1 = X_n + self.dt * (1.5 * f_n - 0.5 * f_nm1)
        self.X_np1 = X_np1

        return X_np1, f_n

    def _step_Y(self, Y_n, g_nm1, X_n):
        """
        Integrate the Y equation in time using Adams-Bashforth

        Parameters
        ----------
        Y_n : (array, size (J,K))
            The small scale variables at time n.
        g_nm1 : (array, size (J, K))
            The right-hand side of Y at time n-1.
            DESCRIPTION.
        X_n : (array, size K)
            The large scale variables at time n.

        Returns
        -------
        Y_np1 : (array, size (J,K))
            The small scale variables at time n+1.
        g_n : (array, size (J,K))
            The right-hand side of Y at time n

        """

        g_n = np.zeros([self.J, self.K])
        for k in range(self.K):
            g_n[:, k] = self.rhs_Y_k(Y_n, X_n[k], k)

        multistep_rhs = self.dt * (1.5 * g_n - 0.5 * g_nm1)

        Y_np1 = Y_n + multistep_rhs

        return Y_np1, g_n

    def step(self, X_n, f_nm1, r_n=None):
        """
        Integrate the L96 system in time. If `r_n` is specified  it is used as 
        a subgrid-scale term. Otherwise the small-scale equation is solved to
        compute the subgrid scale term.

        Parameters
        ----------
        X_n : array (size (K))
            The large scale variables at time n.
        f_nm1 : array (size (K))
            The right-hand side of X at time n-1.
        r_n : array or None
            Use to replace the exact subgrid scale term by a surrogate. 
            Default is None.

        Returns
        -------
        X_np1 : array (size (K))
            The large scale variables at time n+1.
        f_n : array (size (K))
            The right-hand side of X at time n.

        """

        if r_n is None:
            # solve small-scale equation
            self.Y_n, self.g_nm1 = self._step_Y(self.Y_n, self.g_nm1, X_n)

            # compute SGS term
            self.r_n = self.h_x * np.mean(self.Y_n, axis=0)
        else:
            self.r_n = r_n

        # solve large-scale equation
        X_np1, f_n = self._step_X(X_n, f_nm1, self.r_n)

        return X_np1, f_n

    def plot_solution(self):
        """
        Makes a polar plot of the macroscopic solution and the subgrid scale term.

        Returns
        -------
        None.

        """
        # plot results
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        theta = np.linspace(0.0, 2.0 * np.pi, self.K + 1)
        # create a whole in the middle
        ax.set_rorigin(-22)
        # set radial tickmarks
        ax.set_rgrids([-10, 0, 10], labels=['', '', ''])[0][1]
        ax.plot(theta, np.append(self.X_np1, self.X_np1[0]), label='x')
        ax.plot(theta, np.append(self.r_n, self.r_n[0]), label='r')
        ax.legend(loc=1)

        plt.show()
