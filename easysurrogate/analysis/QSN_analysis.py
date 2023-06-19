"""
CLASS TO PERFORM ANALYSIS ON RESULTS FROM A QUANTIZED SOFTMAX SURROGATE.
"""
from .base import BaseAnalysis
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class QSN_analysis(BaseAnalysis):
    """
    QSN analysis class
    """

    def __init__(self, qsn_surrogate, **kwargs):
        print('Creating QSN_analysis object')
        self.qsn_surrogate = qsn_surrogate

    def get_classification_error(self, **kwargs):
        """
        Compute the misclassification error of the QSN surrogate.

        Parameters
        ----------


        Returns
        -------
        Prints classification error to screen for every softmax layer

        """

        if 'X' not in kwargs:
            X = self.qsn_surrogate.neural_net.X
            y = self.qsn_surrogate.neural_net.y
        else:
            X = kwargs['X']
            y = kwargs['y']

        self.qsn_surrogate.neural_net.compute_misclass_softmax(X=X, y=y)

    # def __bin_data(self, data):
    def bin_data(self, data):

        n_bins = self.qsn_surrogate.n_bins
        n_vars = self.qsn_surrogate.feat_eng.n_vars

        n_samples = data.shape[0]

        binnumbers = np.zeros([n_samples, n_vars]).astype('int')
        y_idx = np.zeros([n_samples, n_bins * n_vars])

        for i in range(n_vars):

            bins = self.qsn_surrogate.feat_eng.bins[i]

            _, _, binnumbers[:, i] = \
                stats.binned_statistic(data[:, i], np.zeros(n_samples),
                                       statistic='count', bins=bins)

            unique_binnumber = np.unique(binnumbers[:, i])

            offset = i * n_bins

            for j in unique_binnumber:
                idx = np.where(binnumbers[:, i] == j)
                y_idx[idx, offset + j - 1] = 1.0

        return y_idx

    def get_errors(self, feats, data, relative=True, return_predictions=False):
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
            Also return the train and test predictions.

        Returns
        -------
        err_train, err_test : float
            The training and test errors

        """
        dims = self.qsn_surrogate.get_dimensions()

        # y_idx_train, y_idx_test = self.__bin_data(data)
        y_idx_train = self.bin_data(data[0:dims['n_train']])
        y_idx_test = self.bin_data(data[dims['n_train']:])

        X_train = (feats[0:dims['n_train'], :] - self.qsn_surrogate.feat_mean) / \
            self.qsn_surrogate.feat_std
        X_test = (feats[dims['n_train']:, :] - self.qsn_surrogate.feat_mean) / \
            self.qsn_surrogate.feat_std

        self.qsn_surrogate.neural_net.compute_misclass_softmax(X=X_train, y=y_idx_train)
        self.qsn_surrogate.neural_net.compute_misclass_softmax(X=X_test, y=y_idx_test)

    def get_KL_errors(self, feats, ref_distributions):

        n_samples = feats.shape[0]
        n_softmax = self.qsn_surrogate.n_softmax

        for i in range(n_samples):
            o_i, _, _ = self.qsn_surrogate.neural_net.get_softmax(feats[i].reshape([1, -1]))

            KL_div = np.zeros(n_softmax)
            for j, y_j in enumerate(np.split(ref_distributions[i], n_softmax)):

                idx_gt0 = np.where(y_j > 0.0)[0]
                KL_div[j] = -np.sum(y_j[idx_gt0] * np.log(o_i[j][idx_gt0] / y_j[idx_gt0]))

            print("KL divergence sample %d = %s" % (i, KL_div))

    # def make_movie(self, n_frames=500):
    #     """
    #     Makes a move using the training data. Left subplot shows the evolution of the
    #     kernel-density estimate, and the right subplot show the time series of the data
    #     and the random samples drawm from the kernel-density estimate. Saves the movie
    #     to a .gif file.

    #     Parameters

    #     n_frames (int): default is 500
    #         The number of frames to use in the movie.

    #     Returns: None
    #     """

    #     # get the (normalized, time-lagged) training data from the neural network
    #     X = self.qsn_surrogate.neural_net.X
    #     y = self.qsn_surrogate.neural_net.y

    #     print('===============================')
    #     print('Making movie...')

    #     # list to store the movie frames in
    #     ims = []
    #     fig = plt.figure(figsize=[8, 4])
    #     ax1 = fig.add_subplot(121, xlabel=r'$B_k$', ylabel=r'', yticks=[])
    #     ax2 = fig.add_subplot(122, xlabel=r'time', ylabel=r'$B_k$')
    #     plt.tight_layout()

    #     # number of features
    #     n_feat = X.shape[1]
    #     # number of softmax layers
    #     n_softmax = self.qsn_surrogate.n_softmax

    #     # allocate memory
    #     samples = np.zeros([n_frames, n_softmax])

    #     # make movie by evaluating the network at TRAINING inputs
    #     for i in range(n_frames):

    #         # draw a random sample from the network
    #         o_i, idx_max, _ = self.qsn_surrogate.neural_net.get_softmax(X[i].reshape([1, n_feat]))
    #         self.qsn_surrogate.sampler.resample(idx_max)
    #         if np.mod(i, 100) == 0:
    #             print('i =', i, 'of', n_frames)

    #         # create a single frame, store in 'ims'
    #         plt2 = ax1.plot(range(self.qsn_surrogate.n_bins),
    #                         np.zeros(self.qsn_surrogate.n_bins),
    #                         o_i[0], 'b', label=r'conditional pmf')
    #         plt3 = ax1.plot(y[i][0], 0.0, 'ro', label=r'data')
    #         plt4 = ax2.plot(y[0:i, 0], 'ro', label=r'data')
    #         plt5 = ax2.plot(samples[0:i, 0], 'g', label='random sample')

    #         if i == 0:
    #             ax1.legend(loc=1, fontsize=9)
    #             ax2.legend(loc=1, fontsize=9)

    #         ims.append((plt2[0], plt3[0], plt4[0], plt5[0],))

    #     # make a movie of all frame in 'ims'
    #     im_ani = animation.ArtistAnimation(fig, ims, interval=20,
    #                                        repeat_delay=2000, blit=True)
    #     im_ani.save('kvm.gif')
    #     print('done. Saved movie to qsn.gif')
