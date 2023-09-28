"""
===============================================================================
CLASS FOR THE BINNING SURROGATE PROCEDURE
-------------------------------------------------------------------------------
Reference:
N. Verheul, D. Crommelin,
"Data-driven stochastic representations of unresolved features in
 multiscale models", Communications in Mathematical Sciences, 14, 5, 2016.

Code: W. Edeling
===============================================================================
"""

from itertools import chain, product
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import easysurrogate as es
from tqdm import tqdm


class Resampler:
    """
    ===========================================================================
    Binning surrogate model, resamples reference data
    ---------------------------------------------------------------------------
    Reference:
    N. Verheul, D. Crommelin,
    "Data-driven stochastic representations of unresolved features in
     multiscale models", Communications in Mathematical Sciences, 14, 5, 2016.

    Code: W. Edeling
    ===========================================================================
    """

    def __init__(self, c, r_ip1, N_bins, lags=None, min_count=1,
                 init_feats=True):
        """
        Create a Resampler object

        Parameters
        ----------
        c : array or list of arrays
            Data for each of the conditioning variables. The shape
            can be (n_samples, ), (n_samples, 1) or (n_samples, k), with k > 1.
            In the latter case the k columns will be treated as k separate
            conditioning variables. Hence if

            c = [c1, c2], with shape(c1) = (1000,1) and shape(c2) = (1000, 2),
            a 3D surrogate is created based on 1000 data points.
        r_ip1 : array
            The reference data that must be resampled, condtioned on c. The
            shape can be (n_samples, ), (n_samples, 1) or (n_samples, k).
            In the last case a single r sample consists of k points.

            Note that in this case if c = [c1, c2], for both c1 and c2 we need
            to have shape(c1) = shape(c2) = n_samples * k.
        N_bins : int or list of int
            The number of bins to use per conditioning variable. If an integer
            is specified, the same number of bins are applied to all
            conditioning variables.
        lags : list, optional
            Apply (time) lags to c.
            Example: if c = [c_1, c_2] and lags = [[1], [1, 2]], the first
            conditional vbariable c_1 is lagged by 1 (time) step and the second
            by 1 and 2 (time) steps. The default is None.
        min_count : int, optional
            How many samples a bin must contain to be considered for resampling.
            The default is 1.
        init_feats : boolean, optional
            If time lags are specified, internally store an initial condition
            for the conditioning variables, based on c and lags.
            The default is True.

        Returns
        -------
        None.

        """

        self.name = "Resampler Surrogate"

        if isinstance(c, np.ndarray):
            c = [c]

        # compute N, the size of 1 r sample
        r_dim = r_ip1[0].ndim
        assert r_dim <= 2, "r_ip1 must have shape (n_samples, ) or \
            (n_samples, N) where N is the (flattened) size of 1 r sample"
        if r_dim in (0, 1):
            self.N = 1
        else:
            self.N = r_ip1[0].shape[1]

        # flatten r_ip1, and check if all conditioning variables are
        # of the same size as r_ip1
        r_ip1 = r_ip1.flatten()
        test = np.array([c_i.shape[0] == r_ip1.size for c_i in c])
        assert (test).all(), "conditioning variables must have the same \
        size as r samples: c.shape[0] == r_ip1.flatten().size for all c"

        # create a Feature_Engineering object
        self.feat_eng = es.methods.Feature_Engineering()

        if lags is not None and init_feats:
            # create an initial condition for the conditioning variables
            # using the data and the specified lags
            self.feat_eng.empty_feature_history(lags)
            self.feat_eng.initial_condition_feature_history(c, start=0)
            # set to False, otherwise results will be overwritten by
            # get_training_data
            init_feats = False

        # prepare training data: create one c array and apply (time) lag if
        # lags are specified
        c, r_ip1, _, _ = self.feat_eng.get_training_data(c, r_ip1, lags=lags,
                                                         init_feats=init_feats)

        # total number of conditional variables including lagged terms
        self.N_c = c.shape[1]

        # set the number of bins per conditional variables
        assert isinstance(N_bins, (int, list)), \
        "N_bins must be an integer or a list of integers"

        # apply the same number of bins for all variables
        if isinstance(N_bins, int):
            self.N_bins = [N_bins for i in range(self.N_c)]
        # user-specified list
        else:
            self.N_bins = N_bins
            bin_types = np.array([type(_bin) for _bin in self.N_bins])
            assert (bin_types == int).all(), "N_bins must contain only integers"

        # check that for every conditional variable an N_bins value exists
        assert len(self.N_bins) == self.N_c, \
            "len(N_bins) must equal the number of condtional variables"

        self.r_ip1 = r_ip1
        self.c = c
        self.lags = lags
        self.max_lag = self.feat_eng.max_lag

        # create the N_c dimensional bins
        bins = self.get_bins(self.N_bins)

        # flatten r_ip1 for binned_statistic
        r_ip1 = r_ip1.flatten()
        # count = number of r_ip1 samples in each bin
        # binnumber = bin indices of the r_ip1 samples. A 1D array, no matter N_c
        count, _, binnumber = stats.binned_statistic_dd(
            c, r_ip1, statistic='count', bins=bins)

        # the unique set of binnumers which have at least one r_ip1 sample
        unique_binnumbers = np.unique(binnumber)

        # array containing r_ip1 indices sorted PER BIN
        idx_of_bin = []

        # number of samples in each bin
        size_of_bin = []

        for b in unique_binnumbers:
            # r_ip1 indices of bin b
            tmp = np.where(binnumber == b)[0]

            idx_of_bin.append(tmp)
            size_of_bin.append(tmp.size)

        # make a 1d array of the nested arrays
        idx_of_bin = np.array(list(chain(*idx_of_bin)))
        size_of_bin = np.array(size_of_bin)

        # the starting offset of each bin in the 1D array idx_of_bin
        # offset[unique_binnumbers] will give the starting index in idx_of_bin for
        # each bin. Some entries are zero, which correspond to empty bins (except for
        # the lowest non-empty binnumber)
        offset = np.zeros(unique_binnumbers.size).astype('int')
        offset[1:] = np.cumsum(size_of_bin[0:-1])
        tmp = np.zeros(np.max(unique_binnumbers) + 1).astype('int')
        tmp[unique_binnumbers] = offset
        self.offset = tmp
        # NOTE: offset stores a zero for non empty bins, a dict would be more
        # efficient, but will not allow to simultaneously select multiple
        # offsets using an array of unique binnumbers

        ####################
        #empty bin handling#
        ####################

        # find indices of empty bins
        # Note: if 'full' is defined as 1 or more sample, binnumbers_nonempty is
        # the same as binnumbers_unique
        assert min_count >= 1, "min_count must satisfy min_count >= 1"
        assert isinstance(min_count, int), "min_count must be an integer"
        # N_c dimensional indices of non-empty bins
        x_idx_nonempty = np.where(count >= min_count)
        # Find corresponding 1D binnumbers
        x_idx_nonempty_p1 = [x_idx_nonempty[i] + 1 for i in range(self.N_c)]
        binnumbers_nonempty = np.ravel_multi_index(x_idx_nonempty_p1, [len(b) + 1 for b in bins])
        N_nonempty = binnumbers_nonempty.size

        # mid points of the bins
        x_mid = [0.5 * (bins[i][1:] + bins[i][0:-1]) for i in range(self.N_c)]

        # mid points of the non-empty bins
        midpoints = np.zeros([N_nonempty, self.N_c])
        for i in range(self.N_c):
            midpoints[:, i] = x_mid[i][x_idx_nonempty[i]]

        self.bins = bins
        self.count = count
        self.binnumber = binnumber
        self.binnumbers_nonempty = binnumbers_nonempty
        self.midpoints = midpoints
        self.idx_of_bin = idx_of_bin
        self.unique_binnumbers = unique_binnumbers

        print('Connecting all possible empty bins to nearest non-empty bin...')
        self.fill_in_blanks()
        print('done.')

        # mean r per cell
        self.rmean, _, _ = stats.binned_statistic_dd(c, r_ip1, statistic='mean', bins=bins)

    def check_outliers(self, c_i, binnumbers_i):
        """
        Check which conditional variables c_i fall within empty bins
        and correct their binnumbers_i by projecting to the nearest
        non-empty bin.

        Parameters
        ----------
        c_i : array
            The values of the conditioning variables.
        binnumbers_i : array
            The binnumbers belonging to c_i.

        Returns
        -------
        None.

        """

        # find out how many BINS with outliers there are
        unique_binnumbers_i = np.unique(binnumbers_i)
        idx = np.where(np.in1d(unique_binnumbers_i, self.binnumbers_nonempty) == False)[0]
        N_outlier_bins = idx.size

        if N_outlier_bins > 0:

            # index of outlier SAMPLES in binnumbers_i
            outliers_idx = np.in1d(binnumbers_i, unique_binnumbers_i[idx]).nonzero()[0]
            N_outliers = outliers_idx.size

            # x location of outliers
            x_outliers = np.copy(c_i[outliers_idx])

            # find non-empty bin closest to the outliers
            closest_idx = np.zeros(N_outliers).astype('int')
            for i in range(N_outliers):
                if self.N_c == 1:
                    dist = np.linalg.norm(self.midpoints - x_outliers[i], 2, 1)
                else:
                    dist = np.linalg.norm(self.midpoints - x_outliers[i, :], 2, 1)
                closest_idx[i] = np.argmin(dist)

            binnumbers_closest = self.binnumbers_nonempty[closest_idx]

            check = self.binnumbers_nonempty[closest_idx]
            x_idx_check = np.unravel_index(check, [len(b) + 1 for b in self.bins])
            x_idx_check = [x_idx_check[i] - 1 for i in range(self.N_c)]

            # overwrite outliers in binnumbers_i with nearest non-empty binnumber
            binnumbers_closest = self.binnumbers_nonempty[closest_idx]
            binnumbers_i[outliers_idx] = binnumbers_closest

    def fill_in_blanks(self):
        """
        Create an apriori mapping between every possible bin and the nearest
        non-empty bin. Non-empty bins will link to themselves.


        Returns
        -------
        None.

        """

        bins_padded = []

        # mid points of all possible bins, including ghost bins
        for i in range(self.N_c):
            dx1 = self.bins[i][1] - self.bins[i][0]
            dx2 = self.bins[i][-1] - self.bins[i][-2]

            # pad the beginning and end of current 1D bin with extrapolated values
            bin_pad = np.pad(self.bins[i], (1, 1), 'constant', constant_values=(
                self.bins[i][0] - dx1, self.bins[i][-1] + dx2))
            bins_padded.append(bin_pad)

        # compute the midpoints of the padded bins
        x_mid_pad = [0.5 * (bins_padded[i][1:] + bins_padded[i][0:-1]) for i in range(self.N_c)]
        self.x_mid_pad_tensor = np.array(list(product(*x_mid_pad)))

        # total number bins
        self.max_binnumber = self.x_mid_pad_tensor.shape[0]

        # # slow implementation using for loop
        # mapping = np.zeros(self.max_binnumber).astype('int')
        # for i in tqdm(range(self.max_binnumber)):

        #     # bin is nonempty, just use current idx
        #     if np.in1d(i, self.unique_binnumbers):
        #         mapping[i] = i
        #     # bin is empty, find nearest non-empty bin
        #     else:
        #         binnumbers_i = np.array([i])
        #         self.check_outliers(self.x_mid_pad_tensor[i].reshape([1, self.N_c]), binnumbers_i)
        #         mapping[i] = binnumbers_i[0]

        # fast vectorized implementation
        # all possible bins
        mapping = np.arange(1, self.max_binnumber + 1)
        # update mapping using 1 check_outliers call
        self.check_outliers(self.x_mid_pad_tensor, mapping)
        # if non empty, link to self
        mapping[self.unique_binnumbers] = self.unique_binnumbers

        self.mapping = mapping

    def plot_2D_binning_object(self):
        """
        Plot a visual representation of a 2D binning object. Also shows the mapping
        between empty to nearest non-empty bins.


        Returns
        -------
        None.

        """

        if self.N_c != 2:
            print('Only works for N_c = 2')
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel=r'conditioning variable 1',
                             ylabel=r'conditioning variable 2')

        # plot bins and (c1, c2) which corresponding to a r sample point
        ax.plot(self.c[:, 0], self.c[:, 1], '+', color='lightgray', alpha=0.3)
        ax.vlines(self.bins[0], np.min(self.c[:, 1]), np.max(self.c[:, 1]))
        ax.hlines(self.bins[1], np.min(self.c[:, 0]), np.max(self.c[:, 0]))

        ax.plot(self.x_mid_pad_tensor[:, 0], self.x_mid_pad_tensor[:, 1], 'g+')

        # plot the mapping
        for i in range(self.max_binnumber):
            ax.plot([self.x_mid_pad_tensor[i][0],
                     self.x_mid_pad_tensor[self.mapping[i]][0]],
                    [self.x_mid_pad_tensor[i][1],
                     self.x_mid_pad_tensor[self.mapping[i]][1]], 'b', alpha=0.6)

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.tight_layout()
        plt.show()

    def _feed_forward(self, c_i, n_mc=1):
        """
        The data-driven model to resample data from the conditional
        density r ~ r | c.

        Parameters
        ----------
        c_i : array
            The conditioning variables.
        n_mc : int, optional
            The number of Monte Carlo samples to draw from each bin.
            The default is 1.

        Returns
        -------
        array
            The mean over the n_mc resampled r values.

        """

        # find in which bins the c_i samples fall
        _, _, binnumbers_i = stats.binned_statistic_dd(c_i, np.zeros(self.N), bins=self.bins)

        # static correction for outliers, using precomputed mapping array
        binnumbers_i = self.mapping[binnumbers_i]

        # convert 1D binnumbers_i to equivalent ND indices
        x_idx = np.unravel_index(binnumbers_i, [len(b) + 1 for b in self.bins])
        x_idx = [x_idx[i] - 1 for i in range(self.N_c)]
        x_idx = tuple(x_idx)

        # random integers between 0 and max bin count for each index in binnumbers_i
        I = np.floor(self.count[x_idx].reshape([self.N, 1]) *
                     np.random.rand(self.N, n_mc)).astype('int')

        # the correct offset for the 1D array idx_of_bin
        start = self.offset[binnumbers_i]

        # get random r sample from each bin indexed by binnumbers_i
        r = np.zeros([n_mc, self.N])

        for i in range(n_mc):
            r[i, :] = self.r_ip1[self.idx_of_bin[start + I[:, i]]].reshape([self.N])

        return np.mean(r, 0)

    def predict(self, c, n_mc=1):
        """
        Make a prediction r(c). Here, r is given by Resampler._feed_foward,
        and c are the conditional variables at the current (time) step.

        If time lags are specified, c will be stored internally and a time-
        lagged c array will be passed to Resampler._feed_foward.

        Parameters
        ----------
        feat : array of list of arrays
               The feature array of a list of feature arrays on which to
               evaluate the surrogate.

        Returns
        -------
        array, shape (N,)
            The prediction of the Resampler surrogate.

        """

        if not isinstance(c, list):
            c = [c]

        # make sure all feature vectors have the same ndim.
        # This will raise an error when for instance X1.shape = (10,) and X2.shape = (10, 1)
        ndims = [X_i.ndim for X_i in c]
        assert all([ndim == ndims[0] for ndim in ndims]),\
            "All features must have the same ndim"

        # make sure features are at most two dimensional arrays
        assert ndims[0] <= 2, "Only 1 or 2 dimensional arrays are allowed as features."

        # time-lagged surrogate
        if self.lags is not None:

            # append the current state X to the feature history
            self.feat_eng.append_feat(c)
            feat = self.feat_eng.get_feat_history()

        return self._feed_forward(feat.reshape([1, -1]), n_mc)

    def print_bin_info(self):
        """
        Print some information about the Resampler surrogate to screen.

        Returns
        -------
        None.

        """
        n_bins = np.prod(self.N_bins)
        print('--------------------------------------')
        print('Total number of samples = %d' % self.r_ip1.size)
        print('Total number of bins = %d' % n_bins)
        print('Total number of non-empty bins = %d' % self.binnumbers_nonempty.size)
        print('Percentage filled = %.1f%%' %
              (self.binnumbers_nonempty.size / float(n_bins) * 100, ))
        print('--------------------------------------')

    def get_bins(self, N_bins):
        """
        Compute the uniform bins of the conditional variables in c

        Parameters
        ----------
        N_bins : list of int
            The number of bins per conditioning variable.

        Returns
        -------
        bins : list of arrays
            A list of 1D uniform bins.

        """

        bins = []

        for i in range(self.N_c):
            bins.append(np.linspace(np.min(self.c[:, i]), np.max(self.c[:, i]), N_bins[i] + 1))

        return bins
