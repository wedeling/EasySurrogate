"""
CLASS TO PERFORM ANALYSIS ON RESULTS FROM A CONVERGENT CROSS MAPPING SURROGATE.
"""
from .base import BaseAnalysis
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial import ConvexHull


class CCM_analysis(BaseAnalysis):
    """
    CCM analysis class
    """

    def __init__(self, ccm_surrogate, **kwargs):
        print('Creating CCM_analysis object')
        self.ccm_surrogate = ccm_surrogate

    def auto_correlation_function(self, X, max_lag):
        """
        Compute the autocorrelation of X over max_lag time steps

        Parameters:
            - X (array, size (N,)): the samples from which to compute the ACF
            - max_lag (int): the max number of time steps, determines max
              lead time

        Returns:
            - R (array): array of ACF values
        """
        return super().auto_correlation_function(X, max_lag)

    def cross_correlation_function(self, X, Y, max_lag):
        """
        Compute the crosscorrelation between X and Y over max_lag time steps

        Parameters:
            - X, Y (array, size (N,)): the samples from which to compute the CCF
            - max_lag (int): the max number of time steps, determines max
              lead time

        Returns:
            - C (array): array of CCF values
        """
        return super().cross_correlation_function(X, Y, max_lag)

    def get_pdf(self, X, Npoints=100):
        """
        Computes a kernel density estimate of the samples in X

        Parameters:
            - X (array): the samples
            - Npoints (int, default = 100): the number of points in the domain of X

        Returns:
            - domain (array of size (Npoints,): the domain of X
            - kde (array of size (Npoints,): the kernel-density estimate
        """
        return super().get_pdf(X, Npoints=Npoints)

    def plot_2D_binning_object(self):
        """
        Visual representation of a 2D binning object. Also shows the mapping
        between empty to nearest non-empty bins.

        Returns
        -------
        None.

        """
        if self.ccm_surrogate.n_feats != 2:
            print('Only works for N_c = 2')
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel=r'conditioning variable 1',
                             ylabel=r'conditioning variable 2')

        # plot bins and (c1, c2) which corresponding to a r sample point
        ax.plot(self.ccm_surrogate.feats[:, 0],
                self.ccm_surrogate.feats[:, 1], '+', color='lightgray', alpha=0.3)
        ax.vlines(self.ccm_surrogate.bins[0], np.min(
            self.ccm_surrogate.feats[:, 1]), np.max(self.ccm_surrogate.feats[:, 1]))
        ax.hlines(self.ccm_surrogate.bins[1], np.min(
            self.ccm_surrogate.feats[:, 0]), np.max(self.ccm_surrogate.feats[:, 0]))

        ax.plot(self.ccm_surrogate.x_mid_pad_tensor[:, 0],
                self.ccm_surrogate.x_mid_pad_tensor[:, 1], 'g+')

        # plot the mapping
        for i in range(self.ccm_surrogate.max_binnumber):
            ax.plot([self.ccm_surrogate.x_mid_pad_tensor[i][0],
                     self.ccm_surrogate.x_mid_pad_tensor[self.ccm_surrogate.mapping[i]][0]],
                    [self.ccm_surrogate.x_mid_pad_tensor[i][1],
                     self.ccm_surrogate.x_mid_pad_tensor[self.ccm_surrogate.mapping[i]][1]],
                    'b',
                    alpha=0.6)

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.tight_layout()
        plt.show()

    def plot_2D_shadow_manifold(self):
        """
        Plots the bins on the manifold of the conditioning variables, and
        the corresponding neighborhoods on the shadow manifold. Click on
        manifold bin to show the corresponding shadow bin.

        Returns
        -------
        None.

        """

        if self.ccm_surrogate.n_feats == 1:
            print('Dimension manifold = 1, only works for dimension of 2 or higher')
            return
        elif self.ccm_surrogate.n_feats > 2:
            print('Dimension manifold > 2, will only plot first 2 dimensions')

        print('Click on the bins on the left to show the corresponding shadow bins.')

        fig = plt.figure('manifolds_and_binnumbers', figsize=[8, 4])
        # colors = cycle(['--r', '--g', '--b', '--m', '--k'])
        self.ax1 = fig.add_subplot(121, title='Manifold')
        self.ax2 = fig.add_subplot(122, title='Shawdow manifold')
        self.ax1.set_xlabel(r'$\mathrm{conditioning\;variable\;at\;t-\tau}$')
        self.ax1.set_ylabel(r'$\mathrm{conditioning\;variable\;at\;t}$')
        self.ax2.set_xlabel(r'$\mathrm{shadow\;variable\;at\;t-\tau}$')
        self.ax2.set_ylabel(r'$\mathrm{shadow\;variable\;at\;t}$')

        # plot the samples
        self.ax1.plot(self.ccm_surrogate.feats[:, 0], self.ccm_surrogate.feats[:, 1],
                      '+', color='lightgray', alpha=0.3)
        self.ax2.plot(self.ccm_surrogate.target[:, 0], self.ccm_surrogate.target[:, 1],
                      '+', color='lightgray', alpha=0.3)

        for idx_i in self.ccm_surrogate.sample_idx_per_bin.keys():

            idx = self.ccm_surrogate.sample_idx_per_bin[idx_i]
            points_c = self.ccm_surrogate.feats[idx, 0:2]
            points_r = self.ccm_surrogate.target[idx, 0:2]

            self.plot_local_bin(points_c, idx_i, self.ax1, '--k')
            self.plot_local_bin(points_r, idx_i, self.ax2, '--k')

        plt.tight_layout()
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        """
        Evenht handler for mouse click in the manifold plot

        Parameters
        ----------
        event : button_press_event.

        Returns
        -------
        None.

        """

        if event.inaxes != self.ax1:
            return

        # get the location of the click
        c_i = np.array([event.xdata, event.ydata]).reshape([1, 2])
        # find corresponding bin index
        _, _, binnumber_i = stats.binned_statistic_dd(c_i, np.zeros(1),
                                                      bins=self.ccm_surrogate.bins)

        print('Bin', binnumber_i[0])
        # map empty bins to nearest non-empty bin
        binnumber_i = self.ccm_surrogate.mapping[binnumber_i][0]
        # find the indices of the samples of the selected bin
        idx = self.ccm_surrogate.sample_idx_per_bin[binnumber_i]
        # find the corresponding points of the features and the data
        points_c = self.ccm_surrogate.feats[idx]
        points_r = self.ccm_surrogate.target[idx]

        # highlight the selected bin + the corresponding shadow bin
        self.plot_local_bin(points_c, binnumber_i, self.ax1, 'r', width=4)
        self.plot_local_bin(points_r, binnumber_i, self.ax2, 'r', width=4)

        plt.draw()

    def plot_local_bin(self, points, binnumber, ax, marker, width=2):
        """
        Highlight a bin on the shadow manifold plot

        Parameters
        ----------
        points : points in a given bin
        binnumber : the corresponding binnumber
        ax : matplotlib axis
        marker : matplotlib marker
        width : line width

        Returns
        -------
        None.

        """
        # if  there are 3 or more samples in current bin, plot the
        # convex hull of all samples in this bin
        if points.shape[0] >= 3:
            # marker = next(colors)
            hull = ConvexHull(points)
            ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], marker, linewidth=width)
            ax.plot([points[hull.vertices[0], 0], points[hull.vertices[-1], 0]],
                    [points[hull.vertices[0], 1], points[hull.vertices[-1], 1]], marker, linewidth=width)
        # print(hull.volume)
        #     #plot the binnumber
        #     x_mid = np.mean(points[hull.vertices, :], axis=0)
        #     ax.text(x_mid[0], x_mid[1], str(binnumber))
        # # ax.plot(X[idx_i, 0], X[idx_i, 1], '+')
        # else:
        #     x_mid = np.mean(points, axis=0)
        #     ax.text(x_mid[0], x_mid[1], str(binnumber))

    def compare_convex_hull_volumes(self):
        """
        Display some statistics on the convex hull volumes of the bins
        of the manifold and the shadow manifold.
        """

        self.vols = []
        self.ratio_vols = []

        # total volume of the featue point cloud and the data cloud
        total_vol_c = ConvexHull(self.ccm_surrogate.feats).volume
        total_vol_r = ConvexHull(self.ccm_surrogate.target).volume

        weights = []

        # loop over all non-empty bin indices
        for idx_i in self.ccm_surrogate.sample_idx_per_bin.keys():
            # get feature points ana corresp data points
            idx = self.ccm_surrogate.sample_idx_per_bin[idx_i]
            points_c = self.ccm_surrogate.feats[idx]
            points_r = self.ccm_surrogate.target[idx]

            # if there are enough points to make a ND convex hull
            if points_c.shape[0] >= self.ccm_surrogate.n_feats + 1:
                # Convex hull of local features / data points
                hull_c = ConvexHull(points_c)
                hull_r = ConvexHull(points_r)
                # volumes fractions
                vol_fraction_c = hull_c.volume / total_vol_c
                vol_fraction_r = hull_r.volume / total_vol_r

                weights.append(vol_fraction_c)
                # volume frac and ration of vol fracs
                self.vols.append([vol_fraction_c, vol_fraction_r])
                self.ratio_vols.append(vol_fraction_c / vol_fraction_r)

        avg_ratio = np.mean(self.ratio_vols)

        # weights = weights/np.sum(weights)
        # print(weights)
        # print(np.sum(weights))
        # weighted_ratio = np.mean(weights*self.ratio_vols)
        # self.weights = weights

        print('Average volume ratio binning cell/shadow cell = %.3f' % avg_ratio)
        print('Max volume ratio binning cell/shadow cell = %.3f' % np.max(self.ratio_vols))
        print('min volume ratio binning cell/shadow cell = %.3f' % np.min(self.ratio_vols))
        # print('Weighted average volume ratio binning cell/shadow cell = %.3f' % weighted_ratio)
