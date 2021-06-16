"""
CLASS FOR A CONVERGENT CROSS MAPPING SURROGATE
"""

import numpy as np
from scipy import stats
from itertools import chain, product, cycle
import matplotlib.pyplot as plt
import sys
from scipy.spatial import ConvexHull
import easysurrogate as es


class CCM_Surrogate:

    def __init__(self, **kwargs):
        print('Creating Convergent Cross-Mapping Object')
        self.name = 'CCM Surrogate'

    ############################
    # START COMMON SUBROUTINES #
    ############################

    def train(self, feats, target, N_bins, lags, min_count=0, test_frac=0):

        # spatial dimension, hardcoded to 1 for now
        self.N = 1

        # Feature engineering object
        self.feat_eng = es.methods.Feature_Engineering()

        # number of training samples
        self.n_samples = feats[0].shape[0]
        # compute the size of the training set based on value of test_frac
        self.n_train = np.int(self.n_samples * (1.0 - test_frac))
        print('Using first', self.n_train, 'of', self.n_samples, 'samples to train QSN')

        # list of features
        X = [X_i[0:self.n_train] for X_i in feats]
        # the data
        y = [target[0:self.n_train]]

        print('Creating time-lagged training data...')
        # X_lagged, _ = self.feat_eng.lag_training_data(X, np.zeros(self.n_train), lags)
        # Y_lagged, _ = self.feat_eng.lag_training_data(y, np.zeros(self.n_train), lags)
        X_lagged, _ = self.feat_eng.get_training_data(X, np.zeros(self.n_train), lags=lags)
        Y_lagged, _ = self.feat_eng.get_training_data(y, np.zeros(self.n_train), lags=lags)
        print('done')

        # total number of conditional variables including lagged terms
        self.n_feats = X_lagged.shape[1]

        # number of unique conditioning variables (not including lags)
        self.N_covar = len(lags)

        self.feats = X_lagged
        self.target = Y_lagged
        self.N_bins = N_bins
        self.lags = lags
        self.max_lag = np.max(list(chain(*lags)))
        self.covar = {}

        for i in range(self.N_covar):
            self.covar[i] = []

        bins = self.get_bins(N_bins)

        count, binedges, binnumber = stats.binned_statistic_dd(self.feats, np.zeros(self.n_samples),
                                                               statistic='count', bins=bins)

        # the unique set of binnumers which have at least one r_ip1 sample
        unique_binnumbers = np.unique(binnumber)

        # some scalars
        binnumber_max = np.max(unique_binnumbers)

        # array containing r_ip1 indices sorted PER BIN
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # SHOULD BE A 1D ARRAY, AN ARRAY OF SIZE [MAX(BINNUMBER), MAX(COUNT)]
        # WILL STORE MOSTLY ZEROS IN HIGHER DIMENSIONS, LEADING TO MEMORY FAILURES
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
        # offset[unique_binnumbers] will give the starting index in idex_of_bin for
        # each bin. Some entries are zero, which correspond to empty bins (except for
        # the lowest non-empty binnumber)
        offset = np.zeros(unique_binnumbers.size).astype('int')
        offset[1:] = np.cumsum(size_of_bin[0:-1])
        tmp = np.zeros(binnumber_max + 1).astype('int')
        tmp[unique_binnumbers] = offset
        self.offset = tmp

        ####################
        #empty bin handling#
        ####################

        # find indices of empty bins
        # Note: if 'full' is defined as 1 or more sample, binnumbers_nonempty is
        # the same as binnumbers_unique
        x_idx_nonempty = np.where(count > min_count)
        x_idx_nonempty_p1 = [x_idx_nonempty[i] + 1 for i in range(self.n_feats)]
        binnumbers_nonempty = np.ravel_multi_index(x_idx_nonempty_p1, [len(b) + 1 for b in bins])
        N_nonempty = binnumbers_nonempty.size

        # mid points of the bins
        x_mid = [0.5 * (bins[i][1:] + bins[i][0:-1]) for i in range(self.n_feats)]

        # mid points of the non-empty bins
        midpoints = np.zeros([N_nonempty, self.n_feats])
        for i in range(self.n_feats):
            midpoints[:, i] = x_mid[i][x_idx_nonempty[i]]

        #########################################
        # find all sample indices per bin index #
        #########################################

        sample_idx_per_bin = {}
        for j in unique_binnumbers:
            sample_idx_per_bin[j] = np.where(binnumber == j)[0]

        self.bins = bins
        self.count = count
        self.binnumber = binnumber
        self.binnumbers_nonempty = binnumbers_nonempty
        self.midpoints = midpoints
        self.idx_of_bin = idx_of_bin
        self.unique_binnumbers = unique_binnumbers
        self.sample_idx_per_bin = sample_idx_per_bin

        print('Connecting all possible empty bins to nearest non-empty bin...')
        self.fill_in_blanks()
        print('done.')

        self.init_feature_history(feats)

    def predict(self, X, stochastic=False):

        if not isinstance(X, list):
            X = [X]
        # append the current state X to the feature history
        self.feat_eng.append_feat(X)
        # get the feature history defined by the specified number of time lags.
        # Here, feat is an array with the same size as the neural network input layer
        feat = self.feat_eng.get_feat_history()
        feat = feat.reshape([1, self.n_feats])

        # find in which bins the c_i samples fall
        _, _, binnumbers_i = stats.binned_statistic_dd(feat, np.zeros(self.N), bins=self.bins)

        # static correction for outliers, using precomputed mapping array
        binnumbers_i = self.mapping[binnumbers_i]

        # the neighbors in the selected bin, on the 'C manifold'
        neighbors = self.feats[self.sample_idx_per_bin[binnumbers_i[0]]]

        # the distance from the current point to all neighbors
        dists = np.linalg.norm(neighbors - feat, axis=1)

        # if the current bin does not contain enough points to form a simplex,
        # adjust K.
        if dists.size < self.n_feats + 1:
            K = dists.size  # adjusted K
        else:
            K = self.n_feats + 1  # simplex K (manifold dimension + 1)

        # sort the distances + select the K nearest neighbors
        idx = np.argsort(dists)[0:K]
        simplex_idx = self.sample_idx_per_bin[binnumbers_i[0]][idx]

        if not stochastic:
            # compute the simplex weights w_i
            d_i = dists[idx]
            if d_i[0] == 0:
                u_i = np.zeros(K)
                u_i[0] = 1.0
            else:
                u_i = np.exp(-d_i / d_i[0])
            w_i = u_i / np.sum(u_i)

            # prediction is the weighted sum of correspoding samples
            # on the showdow manifold
            shadow_sample = np.sum(self.target[simplex_idx, -1] * w_i)

            return shadow_sample
        else:
            shadow_sample = self.sample_simplex(self.target[simplex_idx])[0]
            return shadow_sample[-1]

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

    def init_feature_history(self, feats, start=0):
        """
        The features are assumed to be lagged in time. Therefore, the initial
        time-lagged feature vector must be set up. The training data is used
        for this.

        Parameters
        ----------
        + feats : a list of the variables used to construct the time-lagged
        features: [var_1, var_2, ...]. Each var_i is an array such that
        var_i[0] gives the value of var_i at t_0, var_i[1] at t_1 etc.

        + start : the starting index of the training features. Default is 0.

        Returns
        -------
        None.

        """
        for i in range(self.max_lag):
            feat = [X_i[start + i] for X_i in feats]
            self.feat_eng.append_feat(feat)

    def sample_simplex(self, xi, n_mc=1):
        """
        Use an analytical function map to points in the n_xi-dim. hypercube to a
        n_xi-dim simplex with vertices xi_k_jl.

        Source:
            W. Edeling, R. Dwight, P. Cinnella,
            Simplex-stochastic collocation method with improved scalability
            Journal of Computational Physics, 310, 301--328, 2016.

        Parameters
        ----------
        xi: array of floats, shape (n_xi + 1, n_xi)

        n_mc : int, number of samples to draw from inside the simplex
            DESCRIPTION. The default is 1.

        Returns
        -------
        n_mc uniformly distributed points inside the simplex.

        """
        n_xi = xi.shape[1]
        P = np.zeros([n_mc, n_xi])
        for k in range(n_mc):
            # random points inside the hypercube
            r = np.random.rand(n_xi)

            # the term of the map is \xi_k_j0
            sample = np.copy(xi[0])
            for i in range(1, n_xi + 1):
                prod_r = 1.
                # compute the product of r-terms: prod(r_{n_xi-j+1}^{1/(n_xi-j+1)})
                for j in range(1, i + 1):
                    prod_r *= r[n_xi - j]**(1. / (n_xi - j + 1))
                # compute the ith term of the sum: prod_r*(\xi_i-\xi_{i-1})
                sample += prod_r * (xi[i] - xi[i - 1])
            P[k, :] = sample

        return P

    # check which c_i fall within empty bins and correct binnumbers_i by
    # projecting to the nearest non-empty bin
    def check_outliers(self, binnumbers_i, c_i):

        # find out how many BINS with outliers there are
        unique_binnumbers_i = np.unique(binnumbers_i)
        idx = np.where(np.in1d(unique_binnumbers_i, self.binnumbers_nonempty) == False)[0]
        N_outlier_bins = idx.size

        if N_outlier_bins > 0:

            # index of outlier SAMPLES in binnumbers_i
            outliers_idx = np.in1d(binnumbers_i, unique_binnumbers_i[idx]).nonzero()[0]
            N_outliers = outliers_idx.size

#            if self.verbose == True:
#                print(N_outlier_bins, ' bins with', N_outliers ,'outlier samples found')

            # x location of outliers
            x_outliers = np.copy(c_i[outliers_idx])

            # find non-empty bin closest to the outliers
            closest_idx = np.zeros(N_outliers).astype('int')
            for i in range(N_outliers):
                if self.n_feats == 1:
                    dist = np.linalg.norm(self.midpoints - x_outliers[i], 2, 1)
                else:
                    dist = np.linalg.norm(self.midpoints - x_outliers[i, :], 2, 1)
                closest_idx[i] = np.argmin(dist)

            binnumbers_closest = self.binnumbers_nonempty[closest_idx]

            check = self.binnumbers_nonempty[closest_idx]
            x_idx_check = np.unravel_index(check, [len(b) + 1 for b in self.bins])
            x_idx_check = [x_idx_check[i] - 1 for i in range(self.n_feats)]

            # overwrite outliers in binnumbers_i with nearest non-empty binnumber
            binnumbers_closest = self.binnumbers_nonempty[closest_idx]

#            if self.verbose == True:
#                print('Moving', binnumbers_i[outliers_idx], '-->', binnumbers_closest)

            binnumbers_i[outliers_idx] = binnumbers_closest

    def fill_in_blanks(self):
        """
        Create an a-priori mapping between every possible bin and the nearest
        non-empty bin. Non-empty bins will link to themselves.

        Returns
        -------
        None.

        """

        bins_padded = []

        # mid points of all possible bins, including ghost bins
        for i in range(self.n_feats):
            dx1 = self.bins[i][1] - self.bins[i][0]
            dx2 = self.bins[i][-1] - self.bins[i][-2]

            # pad the beginning and end of current 1D bin with extrapolated values
            bin_pad = np.pad(self.bins[i], (1, 1), 'constant', constant_values=(
                self.bins[i][0] - dx1, self.bins[i][-1] + dx2))
            bins_padded.append(bin_pad)

        # compute the midpoints of the padded bins
        x_mid_pad = [0.5 * (bins_padded[i][1:] + bins_padded[i][0:-1]) for i in range(self.n_feats)]
        self.x_mid_pad_tensor = np.array(list(product(*x_mid_pad)))

        # total number bins
        self.max_binnumber = self.x_mid_pad_tensor.shape[0]

        mapping = np.zeros(self.max_binnumber).astype('int')

        for i in range(self.max_binnumber):

            if np.mod(i, 10) == 0:
                progress(i, self.max_binnumber)

            # bin is nonempty, just use current idx
            # if np.in1d(i, self.unique_binnumbers) == True:
            if np.in1d(i, self.binnumbers_nonempty):
                mapping[i] = i
            # bin is empty, find nearest non-empty bin
            else:
                binnumbers_i = np.array([i])
                self.check_outliers(
                    binnumbers_i, self.x_mid_pad_tensor[i].reshape([1, self.n_feats]))
                mapping[i] = binnumbers_i[0]

        self.mapping = mapping

    def print_bin_info(self):
        print('-------------------------------')
        print('Total number of samples= ', self.r_ip1.size)
        print('Total number of non-empty bins = ', self.binnumbers_nonempty.size)
        print('-------------------------------')

    # compute the uniform bins of the conditional variables in c
    def get_bins(self, N_bins):

        bins = []

        for i in range(self.n_feats):
            bins.append(np.linspace(np.min(self.feats[:, i]),
                                    np.max(self.feats[:, i]), N_bins[i] + 1))

        return bins


def progress(count, total, status=''):
    """
    progress bar for the command line
    Source: https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
