"""
CLASS TO PERFORM ANALYSIS ON RESULTS FROM AN ARTIFICIAL NEURAL NETWORK.
"""
import numpy as np
from .base import BaseAnalysis


class ANN_analysis(BaseAnalysis):
    """
    ANN analysis class
    """

    def __init__(self, ann_surrogate):
        print('Creating ANN_analysis object')
        self.ann_surrogate = ann_surrogate

    def sensitivity_measures(self, feats):
        """
        EXPERIMENTAL: Compute global derivative-based sensitivity measures using the
        derivative of squared L2 norm of the output, computing usoing back propagation.
        Integration of the derivatives over the input space is done via MC on the provided
        input features in feats.

        Parameters
        ----------
        feats : array
            An array of input parameter values.

        Returns
        -------
        idx : array
            Indices corresponding to input variables, ordered from most to least
            influential.

        """

        # standardize the features
        feats = (feats - self.ann_surrogate.feat_mean) / self.ann_surrogate.feat_std
        N = feats.shape[0]
        # Set the batch size to 1
        self.ann_surrogate.neural_net.set_batch_size(1)
        # initialize the derivatives
        self.ann_surrogate.neural_net.d_norm_y_dX(feats[0].reshape([1, -1]))
        # compute the squared gradient
        norm_y_grad_x2 = self.ann_surrogate.neural_net.layers[0].delta_hy**2
        # compute the mean gradient
        mean = norm_y_grad_x2 / N
        # loop over all samples
        for i in range(1, N):
            # compute the next (squared) gradient
            self.ann_surrogate.neural_net.d_norm_y_dX(feats[i].reshape([1, -1]))
            norm_y_grad_x2 = self.ann_surrogate.neural_net.layers[0].delta_hy**2
            mean += norm_y_grad_x2 / N
        # order parameters from most to least influential based on the mean
        # squared gradient
        idx = np.fliplr(np.argsort(np.abs(mean).T))
        print('Parameters ordered from most to least important:')
        print(idx)
        return idx, mean

    def print_errors(self, feats, data):
        """
        Print the relative training and test error of the ANN surrogate to screen. This method
        uses the ANN_Surrogate.get_dimensions() dictionary to determine where the split
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

        Returns
        -------
        None.

        """
        dims = self.ann_surrogate.get_dimensions()
        # run the trained model forward at training locations
        n_mc = dims['n_train']
        pred = np.zeros([n_mc, dims['n_out']])
        for i in range(n_mc):
            pred[i, :] = self.ann_surrogate.predict(feats[i])

        train_data = data[0:dims['n_train']]
        rel_err_train = np.linalg.norm(train_data - pred) / np.linalg.norm(train_data)
        print("Training error = %.4f %%" % (rel_err_train * 100))

        # run the trained model forward at test locations
        pred = np.zeros([dims['n_test'], dims['n_out']])
        for idx, i in enumerate(range(dims['n_train'], dims['n_samples'])):
            pred[idx] = self.ann_surrogate.predict(feats[i])
        test_data = data[dims['n_train']:]
        rel_err_test = np.linalg.norm(test_data - pred) / np.linalg.norm(test_data)
        print("Test error = %.4f %%" % (rel_err_test * 100))
