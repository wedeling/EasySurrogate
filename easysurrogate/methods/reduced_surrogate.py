import numpy as np
from ..campaign import Campaign
import easysurrogate as es

"""
===============================================================================
CLASS FOR A REDUCED SURROGATE FOR SPECTRAL MODELS
------------------------------------------------------------------------------
Author: W. Edeling
Source: W. Edeling, D. Crommelin, Reducing data-driven dynamical subgrid scale
models by physical constraints, Computer & Fluids, 2020.
===============================================================================
"""


class Reduced_Surrogate(Campaign):

    def __init__(self, n_qoi, n_model_1d, **kwargs):
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
        V : TYPE
            DESCRIPTION.
        qois_ref : TYPE
            DESCRIPTION.
        qois_model : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if not isinstance(qois_ref, np.ndarray) or not isinstance(qois_model, np.ndarray):
            print('Training requires QoI data stored in numpy array')
            return

        if not qois_ref.size == self.n_qoi or not qois_model.size == self.n_qoi:
            print('size dQ_data must be %d' % self.n_qoi)
            return

        V_hat = np.zeros([self.n_qoi, self.n_model_1d, self.n_model_1d]) + 0.0j
        for i in range(self.n_qoi):
            V_hat[i] = V[i]

        dQ_data = qois_ref - qois_model

        return self.reduced_r(V_hat, dQ_data)

    def predict(self):
        return NotImplementedError

    def save_state(self):
        """
        Save the state of the QSN surrogate to a pickle file
        """
        state = self.__dict__
        super().save_state(state=state, name=self.name)

    def load_state(self):
        """
        Load the state of the QSN surrogate from file
        """
        super().load_state(name=self.name)

    ##########################
    # END COMMON SUBROUTINES #
    ##########################

    def reduced_r(self, V_hat, dQ):
        """
        Compute the reduced SGS term
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

        c_ij = self.compute_cij_using_V_hat(V_hat, inner_prods)

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
            src_Q_i = self.compute_int(V_hat[i], P_hat_i, self.n_model_1d)

            # compute tau_i = Delta Q_i/ (V_i, P_i)
            tau_i = dQ[i] / src_Q_i

            src_Q[i] = src_Q_i
            tau[i] = tau_i

            # compute reduced soure term
            sgs_hat -= tau_i * P_hat_i

        reduced_dict = {'sgs_hat':sgs_hat, 'c_ij':c_ij, 
                        'inner_prods':np.triu(inner_prods),
                        'src_Q':src_Q, 'tau':tau}

        return reduced_dict

    def compute_cij_using_V_hat(self, V_hat, inner_prods):
        """
        compute the coefficients c_ij of P_i = T_{i,1} - c_{i,2}*T_{i,2}, - ...
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
        """
        V_hat = V_hat.reshape([self.n_qoi, self.n_model_1d**2])

        return np.dot(V_hat, np.conjugate(V_hat).T) / self.n_model_1d**4

    def compute_int(self, X1_hat, X2_hat, N):
        """
        Compute the integral of X1*X2 using the Fourier expansion
        """
        integral = np.dot(X1_hat.flatten(), np.conjugate(X2_hat.flatten())) / N**4
        return integral.real
