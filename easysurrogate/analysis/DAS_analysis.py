"""
CLASS TO PERFORM ANALYSIS ON RESULTS FROM A DEEP ACTIVE SUBSPACE NETWORK
"""

import numpy as np
from .base import BaseAnalysis


class DAS_analysis(BaseAnalysis):
    """
    DAS analysis class
    """

    def __init__(self, das_surrogate):
        print('Creating DAS_analysis object')
        self.das_surrogate = das_surrogate

    def sensitivity_measures(self, feats, norm=True):
        """
        Compute global derivative-based sensitivity measures using the
        derivative of squared L2 norm of the output, computing using back propagation.
        Integration of the derivatives over the input space is done via MC on the provided
        input features in feats.

        Parameters
        ----------
        feats : array
            An array of input parameter values.

        norm : Boolean, optional, default is True.
            Compute the gradient of ||y||^2_2. If False it computes the gradient of
            y, if y is a scalar. If False and y is a vector, the resulting gradient is the
            column sum of the full Jacobian matrix.

        Returns
        -------
        idx : array
            Indices corresponding to input variables, ordered from most to least
            influential.

        """

        # standardize the features
        feats = (feats - self.das_surrogate.feat_mean) / self.das_surrogate.feat_std
        print(np.mean(feats, axis=0))
        N = feats.shape[0]
        # Set the batch size to 1
        self.das_surrogate.neural_net.set_batch_size(1)
        mean = 0.0
        # loop over all samples
        for i in range(N):
            # compute the next (squared) gradient
            df_dx = self.das_surrogate.neural_net.d_norm_y_dX(feats[i].reshape([1, -1]),
                                                              norm=norm)
            # norm_y_grad_x2 = self.das_surrogate.neural_net.layers[0].delta_hy**2
            mean += df_dx**2 / N
        # order parameters from most to least influential based on the mean
        # squared gradient
        idx = np.fliplr(np.argsort(mean.T))
        print('Parameters ordered from most to least important:')
        print(idx)
        return idx, mean

    def get_errors(self, feats, data, relative = False):
        """
        Get the training and test error of the DAS surrogate to screen. This method
        uses the DAS_Surrogate.get_dimensions() dictionary to determine where the split
        between training and test data is:

            [0,1,...,n_train,n_train+1,...,n_samples]

        Hence the last entries are used as test data, and feats and data should structured as
        such.

        Parameters
        ----------
        feats : array, size = [n_samples, n_feats]
            The features.
        data : array, size = [n_samples, n_out]
            The data.
        relative: boolean, default is False
            Compute relative instead of absolute errors.

        Returns
        -------
        err_train, err_test : float
            The training and test errors

        """
        dims = self.das_surrogate.get_dimensions()
        # run the trained model forward at training locations
        n_mc = dims['n_train']
        pred = np.zeros([n_mc, dims['n_out']])
        for i in range(n_mc):
            pred[i, :] = self.das_surrogate.predict(feats[i])

        train_data = data[0:dims['n_train']]
        if relative:
            err_train = np.linalg.norm(train_data - pred) / np.linalg.norm(train_data)
            print("Relative training error = %.4f %%" % (err_train * 100))
        else:
            err_train = np.linalg.norm(train_data - pred)
            print("Training error = %.4f " % (err_train))

        # run the trained model forward at test locations
        pred = np.zeros([dims['n_test'], dims['n_out']])
        for idx, i in enumerate(range(dims['n_train'], dims['n_samples'])):
            pred[idx] = self.das_surrogate.predict(feats[i])
        test_data = data[dims['n_train']:]
        if relative:
            err_test = np.linalg.norm(test_data - pred) / np.linalg.norm(test_data)
            print("Relative test error = %.4f %%" % (err_test * 100))
        else:
            err_test = np.linalg.norm(test_data - pred)

        return err_train, err_test
