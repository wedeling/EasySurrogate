"""
CLASS TO PERFORM ANALYSIS ON RESULTS FROM A KERNEL MIXTURE SURROGATE.
"""
from .base import BaseAnalysis
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import animation
# from sklearn.neighbors import KernelDensity


class KMN_analysis(BaseAnalysis):
    """
    KMN analysis class
    """

    def __init__(self, kmn_surrogate, **kwargs):
        print('Creating KMN_analysis object')
        self.kmn_surrogate = kmn_surrogate

    def compute_kde(self, dom, w, mu, sigma):
        """
        Compute a kernel density estimate (KDE) given the weights w, which are the output
        of the KMN surrogate.

        Parameters:
            - dom (array): domain of the KDE
            - w (array): weights of the KDE, as predicted by the network. Sum to 1.
        """

        K = norm.pdf(dom, mu, sigma)
        w = w.reshape([w.size, 1])

        return np.sum(w * K, axis=0)

    def make_movie(self, n_frames=500):
        """
        Makes a move using the training data. Left subplot shows the evolution of the
        kernel-density estimate, and the right subplot show the time series of the data
        and the random samples drawm from the kernel-density estimate. Saves the movie
        to a .gif file.

        Parameters

        n_frames (int): default is 500
            The number of frames to use in the movie.

        Returns: None
        """

        # get the (normalized, time-lagged) training data from the neural network
        X = self.kmn_surrogate.neural_net.X
        y = self.kmn_surrogate.neural_net.y

        print('===============================')
        print('Making movie...')

        # list to store the movie frames in
        ims = []
        fig = plt.figure(figsize=[8, 4])
        ax1 = fig.add_subplot(121, xlabel=r'$B_k$', ylabel=r'', yticks=[])
        ax2 = fig.add_subplot(122, xlabel=r'time', ylabel=r'$B_k$')
        plt.tight_layout()

        # number of features
        n_feat = X.shape[1]
        # number of softmax layers
        n_softmax = self.kmn_surrogate.n_softmax
        # kernel properties, mean and standard dev
        mu = self.kmn_surrogate.kernel_means
        sigma = self.kmn_surrogate.kernel_stds
        # number of points to use for plotting the conditional pdfs
        n_kde = 100

        # domain of the conditional pdfs
        dom = np.linspace(np.min(y), np.max(y), n_kde)

        # allocate memory
        kde = np.zeros([n_kde, n_softmax])
        samples = np.zeros([n_frames, n_softmax])

        # make movie by evaluating the network at TRAINING inputs
        for i in range(n_frames):

            # draw a random sample from the network
            w, idx_max, _ = self.kmn_surrogate.neural_net.get_softmax(X[i].reshape([1, n_feat]))
            for j in range(n_softmax):
                kde[:, j] = self.compute_kde(dom, w[j], mu[j], sigma[j])
                samples[i, j] = norm.rvs(mu[j][idx_max[j]], sigma[j][idx_max[j]])
            if np.mod(i, 100) == 0:
                print('i =', i, 'of', n_frames)

            # create a single frame, store in 'ims'
            plt2 = ax1.plot(dom, kde[:, 0] / np.max(kde[:, 0]), 'b', label=r'conditional pdf')
            plt3 = ax1.plot(y[i][0], 0.0, 'ro', label=r'data')
            plt4 = ax2.plot(y[0:i, 0], 'ro', label=r'data')
            plt5 = ax2.plot(samples[0:i, 0], 'g', label='random sample')

            if i == 0:
                ax1.legend(loc=1, fontsize=9)
                ax2.legend(loc=1, fontsize=9)

            ims.append((plt2[0], plt3[0], plt4[0], plt5[0],))

        # make a movie of all frame in 'ims'
        im_ani = animation.ArtistAnimation(fig, ims, interval=20,
                                           repeat_delay=2000, blit=True)
        im_ani.save('kvm.gif')
        print('done. Saved movie to kvm.gif')
