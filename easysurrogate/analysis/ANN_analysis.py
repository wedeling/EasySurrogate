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

    def sensitivity_measures(self, feats, norm=True):
        """
        Compute global derivative-based sensitivity measures using the
        derivative of squared L2 norm of the output, computing usoing back propagation.
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
        feats = (feats - self.ann_surrogate.feat_mean) / self.ann_surrogate.feat_std
        N = feats.shape[0]
        # Set the batch size to 1
        self.ann_surrogate.neural_net.set_batch_size(1)
        # initialize the derivatives
        self.ann_surrogate.neural_net.d_norm_y_dX(feats[0].reshape([1, -1]),
                                                  norm=norm)
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

    def get_errors(self, feats, data, relative = True, return_predictions=False):
        """
        Get the training and test error of the ANN surrogate to screen. This method
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
        relative : boolean, default is True
            Compute relative instead of absolute errors.
        return_predictions : boolean, default is False
            Also return the traiin and test predictions.

        Returns
        -------
        err_train, err_test : float
            The training and test errors

        """
        dims = self.ann_surrogate.get_dimensions()
        # run the trained model forward at training locations
        n_mc = dims['n_train']
        train_pred = np.zeros([n_mc, dims['n_out']])
        for i in range(n_mc):
            train_pred[i, :] = self.ann_surrogate.predict(feats[i])

        train_data = data[0:dims['n_train']]
        if relative:
            err_train = np.mean(np.linalg.norm(train_data - train_pred, axis=0) /
                                np.linalg.norm(train_data, axis=0), axis=0)
            print("Relative training error = %.4f %%" % (err_train * 100))
        else:
            err_train = np.mean(np.linalg.norm(train_data - train_pred, axis=0), axis=0)
            print("Training error = %.4f " % (err_train))

        # run the trained model forward at test locations
        test_pred = np.zeros([dims['n_test'], dims['n_out']])
        for idx, i in enumerate(range(dims['n_train'], dims['n_samples'])):
            test_pred[idx] = self.ann_surrogate.predict(feats[i])
        test_data = data[dims['n_train']:]
        if relative:
            err_test = np.mean(np.linalg.norm(test_data - test_pred, axis=0) /
                               np.linalg.norm(test_data,axis=0), axis=0)
            print("Relative test error = %.4f %%" % (err_test * 100))
        else:
            err_test = np.mean(np.linalg.norm(test_data - test_pred, axis=0), axis=0)
            print("Test error = %.4f " % (err_test))

        if not return_predictions:
            return err_train, err_test
        else:
            return err_train, err_test, train_pred, test_pred
