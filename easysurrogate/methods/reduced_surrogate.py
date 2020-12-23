"""
===============================================================================
CLASS FOR A REDUCED SURROGATE FOR SPECTRAL MODELS
------------------------------------------------------------------------------
Author: W. Edeling
Source: W. Edeling, D. Crommelin, Reducing data-driven dynamical subgrid scale
models by physical constraints, Computer & Fluids, 2020.
===============================================================================
"""

import numpy as np
from ..campaign import Campaign


class Reduced_Surrogate(Campaign):
    """
    Reduced Surrogate class
    """

    def __init__(self, n_qoi, n_model_1d):
        """
        Create a Reduced Surrogate object.

        Parameters
        ----------
        n_qoi : int
            The number of integrated QoI to track.
        n_model_1d : int
            The number of grid points in one direction.

        Returns
        -------
        None.

        """
        self.n_qoi = n_qoi
        self.n_model_1d = n_model_1d
        print('Creating Reduced Surrogate Object')
        self.name = 'Reduced Surrogate'

    ############################
    # START COMMON SUBROUTINES #
    ############################

    def train(self, V, qois_ref, qois_model):
        """

        Parameters
        ----------
        V : list
            A list containing the Fourier coefficients of reduced basis functions V_i.
        qois_ref : array
            The reference statistics at a given time step.
        qois_model : array
            The same statistics computed by the lower-resolution model.

        Returns
        -------
        dict
            A dictionary containing:
            'sgs_hat': the reduced subgrid-scale term,
            'c_ij': the coefficients of the linear system A_i * c_i = b_i (Eq 20-21 in the paper),
            'inner_prods': the inner products (V_i, T_{i,j}) (Eq 21),
            'src_Q': the inner product (V_i, P_i) that is part of a source term of Q (Eq 23),
            'tau': the multiplier of (V_i, P_i) (Eq 23).

        """

        assert isinstance(qois_ref, np.ndarray) and isinstance(qois_model, np.ndarray), \
            'Training requires QoI data stored in numpy array'

        assert qois_ref.size == self.n_qoi and qois_model.size == self.n_qoi, \
            'size dQ_data must be %d' % self.n_qoi

        V_hat = np.zeros([self.n_qoi, self.n_model_1d, self.n_model_1d]) + 0.0j
        for i in range(self.n_qoi):
            V_hat[i] = V[i]

        dQ_data = qois_ref - qois_model

        return self.reduced_r(V_hat, dQ_data)

    def predict(self, V, features, **kwargs):
        """
        Predict using a reduced subgrid-scale term, with a surrogate for dQ := qois_ref - qois_model

        Parameters
        ----------
        V : list
            A list containing the Fourier coefficients of reduced basis functions V_i.
        features : list or array
            A single feature array ot a list of multiple feature arrays.

        Returns
        -------
        dict
            A dictionary containing:
            'sgs_hat': the reduced subgrid-scale term,
            'c_ij': the coefficients of the linear system A_i * c_i = b_i (Eq 20-21 in the paper),
            'inner_prods': the inner products (V_i, T_{i,j}) (Eq 21),
            'src_Q': the inner product (V_i, P_i) that is part of a source term of Q (Eq 23),
            'tau': the multiplier of (V_i, P_i) (Eq 23).

        """

        # add all V_i into a single array
        V_hat = np.zeros([self.n_qoi, self.n_model_1d, self.n_model_1d]) + 0.0j
        for i in range(self.n_qoi):
            V_hat[i] = V[i]

        # predict dQ := qois_ref - qoi_model using an EasySurrogate surrogate
        if 'dQ' not in kwargs:
            if not hasattr(self, 'dQ_surr'):
                print('Reduced_Surrogate object does not have a surrogate for dQ:')
                print('use set_dQ_surrogate subroutine.')
                return
            dQ_surr = self.dQ_surr.predict(features)
        else:
            dQ_surr = kwargs['dQ']

        return self.reduced_r(V_hat, dQ_surr)

    def save_state(self):
        """
        Save the state of the reduced surrogate to a pickle file.

        Returns
        -------
        None.

        """

        state = self.__dict__
        super().save_state(state=state, name=self.name)

    def load_state(self):
        """
        Load the state of the reduced surrogate from file.

        Returns
        -------
        None.

        """
        super().load_state(name=self.name)

    ##########################
    # END COMMON SUBROUTINES #
    ##########################

    def set_dQ_surrogate(self, surrogate):
        """
        Set the surrogate model for dQ := Q_reference - Q_model

        Parameters
        ----------
        surrogate : object
            An EasySurrogate method trained to predict dQ.

        Returns
        -------
        None.

        """
        self.dQ_surr = surrogate

    def reduced_r(self, V_hat, dQ):
        """
        Compute the reduced SGS term

        Parameters
        ----------
        V_hat : list
            A list containing the Fourier coefficients of reduced basis functions V_i.
        dQ : array
            The difference in the QoI between the high and low resolution model.

        Returns
        -------
        reduced_dict : dict
            A dictionary containing:
            'sgs_hat': the reduced subgrid-scale term,
            'c_ij': the coefficients of the linear system A_i * c_i = b_i (Eq 20-21 in the paper),
            'inner_prods': the inner products (V_i, T_{i,j}) (Eq 21),
            'src_Q': the inner product (V_i, P_i) that is part of a source term of Q (Eq 23),
            'tau': the multiplier of (V_i, P_i) (Eq 23).
        """

        # compute the T_ij basis functions
        T_hat = np.zeros([self.n_qoi, self.n_qoi, self.n_model_1d, self.n_model_1d]) + 0.0j

        for i in range(self.n_qoi):

            T_hat[i, 0] = V_hat[i]

            J = np.delete(np.arange(self.n_qoi), i)

            idx = 1
            for j in J:
                T_hat[i, idx] = V_hat[j]
                idx += 1

        # compute the coefficients c_ij
        inner_prods = self.inner_products(V_hat)

        c_ij = self.compute_cij_using_V_hat(inner_prods)

        sgs_hat = 0.0

        src_Q = np.zeros(self.n_qoi)
        tau = np.zeros(self.n_qoi)

        # loop over all QoI
        for i in range(self.n_qoi):
            # compute the fourier coefs of the P_i
            P_hat_i = T_hat[i, 0]
            for j in range(0, self.n_qoi - 1):
                P_hat_i -= c_ij[i, j] * T_hat[i, j + 1]

            # (V_i, P_i) integral
            src_Q_i = compute_int(V_hat[i], P_hat_i, self.n_model_1d)

            # compute tau_i = Delta Q_i/ (V_i, P_i)
            tau_i = dQ[i] / src_Q_i

            src_Q[i] = src_Q_i
            tau[i] = tau_i

            # compute reduced soure term
            sgs_hat -= tau_i * P_hat_i

        reduced_dict = {'sgs_hat': sgs_hat, 'c_ij': c_ij,
                        'inner_prods': np.triu(inner_prods),
                        'src_Q': src_Q, 'tau': tau}

        return reduced_dict

    def compute_cij_using_V_hat(self, inner_prods):
        """
        Compute the coefficients c_ij of P_i = T_{i,1} - c_{i,2}*T_{i,2}, - ... (Eq 16, 20 and 21)

        Parameters
        ----------
        inner_prods : array
            A matrix containing all inner products (V_i, T_{i,j}).

        Returns
        -------
        c_ij : array
            The coefficients of the subgrid-scale basis functions (Eq 16, 20 and 21) at the
            current time step.

        """
        c_ij = np.zeros([self.n_qoi, self.n_qoi - 1])

        for i in range(self.n_qoi):
            A = np.zeros([self.n_qoi - 1, self.n_qoi - 1])
            b = np.zeros(self.n_qoi - 1)

            k = np.delete(np.arange(self.n_qoi), i)

            for j1 in range(self.n_qoi - 1):
                for j2 in range(j1, self.n_qoi - 1):
                    A[j1, j2] = inner_prods[k[j1], k[j2]]
                    if j1 != j2:
                        A[j2, j1] = A[j1, j2]

            for j1 in range(self.n_qoi - 1):
                b[j1] = inner_prods[i, k[j1]]

            if self.n_qoi == 2:
                c_ij[i, :] = b / A
            else:
                c_ij[i, :] = np.linalg.solve(A, b)

        return c_ij

    def inner_products(self, V_hat):
        """
        Compute all the inner products (V_i, T_{i,j})

        Parameters
        ----------
        V_hat : list
            A list containing the Fourier coefficients of reduced basis functions V_i.

        Returns
        -------
        inner_prods : array
            A matrix containing all inner products (V_i, T_{i,j}).

        """
        V_hat = V_hat.reshape([self.n_qoi, self.n_model_1d**2])

        # TODO: I added the .real to get rid of a warning. Should be ok, but check if this does
        # not yield problems
        return np.dot(V_hat, np.conjugate(V_hat).T).real / self.n_model_1d**4


def compute_int(X1_hat, X2_hat, N):
    """
    Compute the integral of X1*X2 over the spatial domain using the Fourier expansion,
    see section 4.4 of the paper.

    Parameters
    ----------
    X1_hat : array (complex)
        The Fourier coefficients of X1.
    X2_hat : array (complex)
        The Fourier coefficients of X2.
    N : int
        The number of points in one spatial direction.

    Returns
    -------
    float
        The value of the integral.

    """
    integral = np.dot(X1_hat.flatten(), np.conjugate(X2_hat.flatten())) / N**4
    return integral.real
