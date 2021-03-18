"""
CLASS TO PERFORM ANALYSIS ON RESULTS FROM A QUANTIZED SOFTMAX SURROGATE.
"""
from .base import BaseAnalysis
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

    def get_classification_error(self, index=None, **kwargs):
        """
        Compute the misclassification error of the QSN surrogate.

        Parameters
        ----------

        index : array, default is None
            indices to select a subset of feature/data points to perform test on. When None
            the classification error is computed on the entire dataset.

        Returns
        -------
        Prints classification error to screen for every softmax layer

        """

        if index is not None:
            X = self.qsn_surrogate.neural_net.X[index]
            y = self.qsn_surrogate.neural_net.y[index]
        else:
            X = self.qsn_surrogate.neural_net.X
            y = self.qsn_surrogate.neural_net.y

        self.qsn_surrogate.neural_net.compute_misclass_softmax(X=X, y=y)

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
