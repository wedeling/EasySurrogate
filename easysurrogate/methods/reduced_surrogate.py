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
import easysurrogate as es
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

    def train(self, V, dQ):
        """
        A training step of the reduced surrogate.

        Parameters
        ----------
        V : list
            A list containing the Fourier coefficients of reduced basis functions V_i.
        dQ : array
            The reference statistics at a given time step - the same statistics computed
            with the lower resolution model.

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

        assert isinstance(dQ, np.ndarray), 'dQ must be of type numpy.ndarray'

        assert dQ.size == self.n_qoi, 'size dQ_data must be %d' % self.n_qoi

        V_hat = np.zeros([self.n_qoi, self.n_model_1d, self.n_model_1d]) + 0.0j
        for i in range(self.n_qoi):
            V_hat[i] = V[i]

        return self.reduced_r(V_hat, dQ)

    def generate_online_training_data(self, feats, LR_before, LR_after, HR_before, HR_after,
                                      qoi_func, **kwargs):
        """
        Compute the features and the target data for an online training step. Results are
        stored internally, and used within the 'train_online' subroutine.

        Source:
        Rasp, "Coupled online learning as a way to tackle instabilities and biases
        in neural network parameterizations: general algorithms and Lorenz 96
        case study", 2020.

        Parameters
        ----------
        feats : array or list of arrays
            The input features
        LR_before : array or list or arrays
            Low resolution state(s) at previous time step.
        LR_after : array or list or arrays
            Low resolution state(s) at current time step.
        HR_before : array or list or arrays
            High resolution state(s) at previous time step.
        HR_after : array or list or arrays
            High resolution state(s) at current time step.
        qoi_func : function
            A user-specfied function f(state, **kwargs) what computes the QoI from the LR
            or HR state.

        Returns
        -------
        None.

        """

        # multiple features arrays are stored in a list. For consistency put a single
        # array also in a list.
        if isinstance(feats, np.ndarray):
            feats = [feats]

        # allow the state to be stored in a list in case there are multiple states
        if isinstance(LR_before, np.ndarray):
            LR_before = [LR_before]
        if isinstance(LR_after, np.ndarray):
            LR_after = [LR_after]
        if isinstance(HR_before, np.ndarray):
            HR_before = [HR_before]
        if isinstance(HR_after, np.ndarray):
            HR_after = [HR_after]

        # store input features
        for i in range(self.n_feat_arrays):
            self.dQ_surr.feat_eng.online_feats[i].append(feats[i])

        dQ = []
        n_HR = HR_before[0].shape[0]
        # loop over all states
        for i in range(len(LR_before)):

            # project the low-res model to the high-res grid
            LR_before_projected = self.up_scale(LR_before[i], n_HR)

            # the difference between the low res and high res model (projected to low-res grid)
            # at time n
            delta_nudge = LR_before_projected - HR_before[i]

            # the estimated state of the (projected) HR model would there have been no nudging
            HR_no_nudge = HR_after[i] - delta_nudge / self.tau_nudge * self.dt_LR

            # compute the HR QoI
            Q_HR = qoi_func(HR_no_nudge, **kwargs)
            # Q_HR = qoi_func(HR_after[i], **kwargs)

            # compute the LR QoI
            # Note: could store multiple functions in a list if required.
            Q_LR = qoi_func(LR_after[i], **kwargs)

            dQ.append(Q_HR - Q_LR)

        # The difference in HR and LR QoI is the target of the surrogate
        self.dQ_surr.feat_eng.online_target.append(np.concatenate(dQ))

        # remove oldest item from the online features is the window lendth is exceeded
        if len(self.dQ_surr.feat_eng.online_feats[0]) > self.window_length:
            for i in range(self.n_feat_arrays):
                self.dQ_surr.feat_eng.online_feats[i].pop(0)
            self.dQ_surr.feat_eng.online_target.pop(0)

    def set_online_training_parameters(self, tau_nudge, dt_LR, window_length):
        """
        Stores parameters required for online training.

        Parameters
        ----------
        tau_nudge : float
            Nudging time scale.
        dt_LR : float
            Time step low resolution model.
        window_length : int
            The length of the moving window in which online features are stored.

        Returns
        -------
        None.

        """
        assert hasattr(self, 'dQ_surr'), "A surrogate for dQ must be set using set_dQ_surr(...)"
        # store the parameters in the Feature Engineering object of the dQ surrogate
        self.dQ_surr.feat_eng.set_online_training_parameters(tau_nudge, dt_LR, window_length)
        # als store it locally
        self.tau_nudge = tau_nudge
        self.dt_LR = dt_LR
        self.window_length = window_length
        # the number of feature vectors for the dQ surrogate
        self.n_feat_arrays = self.dQ_surr.feat_eng.n_feat_arrays

    def predict(self, V, dQ_surr):
        """
        Compute the reduced surrogate subgrid-scale term using a surrogate for dQ := qoi_ref -
        qoi_model

        Parameters
        ----------
        V : list
            A list containing the Fourier coefficients of reduced basis functions V_i.
        dQ_surr : array
            A surrogate prediction of dQ.

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
        # the code is acutally the same as in the training step, only dQ is computed using a
        # surrogate, instead of extracted from the high-resolution model.
        return self.train(V, dQ_surr)

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

    def set_dQ_surr(self, surrogate):
        """
        Set the surrogate used for predicting dQ := Q_reference - Q_model

        Parameters
        ----------
        surrogate : object
            A surrogate trained on dQ data.

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

            # compute reduced subgrid-scale source term
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

        return np.dot(V_hat, np.conjugate(V_hat).T).real / self.n_model_1d**4

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
        # of X_hat
        for i in range(X_hat.ndim):
            X_hat = np.delete(X_hat, np.arange(start, end), axis=i)
        # In numpy only the inverse transform is scaled. The following term must be applied
        # to ensure the correct scaling factor is applied in the inverse transform.
        scaling_factor = (N_LR / N_HR)**d
        return X_hat * scaling_factor

    def up_scale(self, X_hat, N_HR):
        """
        Up-scale X to a higher spatial resolution by padding high-frequency Fourier coefficients with
        zeros. Thus far this will only work for 1 or 2 dimensional arrays.


        Parameters
        ----------
        X_hat : array (complex)
            The Fourier coefficients of X.
        N_HR : int
            The new, higher, spatial resolution (X.shape[0] > N).

        Returns
        -------
        X_hat : array (complex)
            The Fourier coefficients of X at a higher spatial resolution determined ny N.

        """

        N_LR = X_hat.shape[0]
        d = X_hat.ndim

        # assert N_LR < N_HR, "X_hat.shape[0] must be < N_HR in order to upscale X_hat."
        assert d == 1 or d == 2, "Upscaling only implemented for 1d or 2d arrays."

        start = int(N_LR / 2)
        pad_size = N_HR - N_LR
        if X_hat.ndim == 1:
            # pad the 1d array with zeros
            X_hat = np.insert(X_hat, start, np.zeros(pad_size) + 0j)
        elif X_hat.ndim == 2:
            # pad the 2d array with a 'cross' of zeros
            X_hat = np.insert(X_hat, start, np.zeros([pad_size, N_LR]) + 0j, axis=0)
            X_hat = np.insert(X_hat, start, np.zeros([pad_size, N_HR]) + 0j, axis=1)
        # In numpy only the inverse transform is scaled. The following term must be applied
        # to ensure the correct scaling factor is applied in the inverse transform.
        scaling_factor = (N_HR / N_LR)**d
        return X_hat * scaling_factor


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
