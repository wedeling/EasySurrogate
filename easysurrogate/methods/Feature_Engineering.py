"""
===============================================================================
CLASS FOR FEATURE ENGINEERING SUBROUTINES
===============================================================================
"""

import numpy as np
from itertools import chain
from scipy import stats

from scipy.optimize import minimize
import statistics


class Feature_Engineering:
    """
    Feature_Engineering class. containing several generic methods for the manipulation of features
    that are shared by all surrogate methods.
    """

    def __init__(self):
        print('Creating Feature Engineering object')
        self.lags = None
        self.local = False

    def _predict(self, X, feed_forward):
        """
        Contains the generic processing of features that is independent of the chosen surrogate
        method. Features are processed depending upon the presence of time lags or the local /
        non-local nature of the surrogate. The processed features are then passed to the
        surrogate-specific feed_forward method.

        Parameters
        ----------
        X : array or list of arrays
            The feature array or list of feature arrays on which to evaluate the surrogate.
        feed_forward : function
            The prediction function that is specific to a particular surrogate.

        Returns
        -------
        array
        feed_forward(X)

        """

        if not isinstance(X, list):
            X = [X]

        # make sure all feature vectors have the same ndim.
        # This will raise an error when for instance X1.shape = (10,) and X2.shape = (10, 1)
        ndims = [X_i.ndim for X_i in X]
        assert all([ndim == ndims[0] for ndim in ndims]), "All features must have the same ndim"

        # make sure features are at most two dimensional arrays
        assert ndims[0] <= 2, "Only 1 or 2 dimensional arrays are allowed as features."

        # in the case of two dimensional arrays, make are the second dimension is the same
        # for all features. This dimension must equal the grid size
        local = False
        if ndims[0] == 2:
            local = True
            shapes1 = [X_i.shape[1] for X_i in X]
            assert all([shape1 == shapes1[0] for shape1 in shapes1]), \
                "The size of the second dimension must be the same for all features."
            # if a second dimension is specified, it is assumed that we must loop over this
            # dimension in order to make a single prediction
            n_points = shapes1[0]

        # time-lagged surrogate
        if self.lags is not None:

            # append the current state X to the feature history
            self.append_feat(X)

            # if not local, get entire feature vector and feed forward
            if not local:
                feat = self.get_feat_history()
                return feed_forward(feat)
            # if local, loop over the 2nd dimension of the feature vector and feed forward
            # every entry
            else:
                y = []
                for p in range(n_points):
                    feat = self.get_feat_history(index=p)
                    y.append(feed_forward(feat))
                return np.array(y).flatten()
        # no lags
        else:
            # no lags and non local, create a single vector of X and feed forward
            if not local:
                feat = np.concatenate(X)
                return feed_forward(feat)
            # no lags and local, get feature vector and loop over 2nd dimension
            else:
                # if passed list of features [n_samples x n_grid_points]
                X = np.array(X).reshape([n_points, len(X)])
                #X_train.append(np.moveaxis(np.array(X[i]), 0, -1).reshape([self.n_train, -1]))
                y = []
                for p in range(n_points):
                    y.append(feed_forward(X[p]))  # GP case: for single sample should be one point
                return np.array(y).flatten()

    def filter_values(self, feats, interval):
        """
        Choose only those samples for which feature value lies
        Args:
            feats:
            interval:

        Returns:

        """

    def chose_feature_from_acquisition(self, acquisition_function, X_cands,
                                       candidate_search=True, n_new_cands=1):
        """
        Returns a new parameter value as a minimum of acquisition function, as well as its index among suggested
        candidates index in surrogate test set.

        1) pass the bounds or better set of candidate points in X (by default: set of test X)
        2) pass the criteria / acquisition function f:X->R
        3) pass the number of new candidates per iteration (default=1)
        4) choose the x_star as argmin acquisition
        5) extrude x_star from test set and add to train set
        6) assign new x_train and x_test
        7) return the new sample to add to training set x_min

                Parameters
        ----------
        acquisition_function : the function which minimum defines the optimal new sample to add to training set
            A callable which can accept a single argument with in same format as .predict() method
        X_cands : list of input parameters / features
            The new sample for training will be chosen from this list
        candidate_search: boolean, if True search among the list of candidate values,
            if False generate new value within boundary box as minimum using scipy.minimize()
        n_new_cands: integer, number of new candidate input points to return

        Returns
        -------
        array
        feed_forward(X)
        """

        if candidate_search:

            cand_vals = [acquisition_function(x) for x in X_cands]
            x_min_ind_test = np.argmin(np.array(cand_vals))
            x_min = X_cands[x_min_ind_test]
            x_min_ind_glob = self.test_indices[x_min_ind_test]

        else:
            boundminima = np.array(X_cands).min(axis=0)
            boundmaxima = np.array(X_cands).max(axis=0)
            currbounds = []
            for i in range(self.n_dim):
                currbounds.append((boundminima[i], boundmaxima[i]))

            opt_start_point = [statistics.mean(x) for x in currbounds]

            newpoints = minimize(acquisition_function, np.array(opt_start_point),
                                 bounds=currbounds)  # bounds for current GP case
            if newpoints.success:
                x_min = newpoints['x']

            x_min_ind_test = 0
            x_min_ind_glob = 0

        print('Using new %d samples to retrain the ML model' % n_new_cands)

        return x_min, x_min_ind_test, x_min_ind_glob

    def get_training_data(
            self,
            feats,
            target,
            lags=None,
            local=False,
            test_frac=0.0,
            valid_frac=0.0,
            train_first=True,
            index=None):
        """
        Generate training data. Training data can be made (time) lagged and/or local.

        Parameters
        ----------
        feats : Array or list of arrays
            A single feature array or a list of different feature arrays. The shape of the feature
            arrays must be (n_samples, n_points_i). Here n_samples is the number of samples and
            n_points_i is the size of a single sample of the i-th feature.
        target : Array
            The target that must be predicted by the surrogate. The shape of this array must be
            (n_samples, n_target), where n_target is the size of a single sample.
        lags : list of lists, optional
            In the case of a time-lagged surrogate, this list contains the time lags of
            each distinct feature. Each time lag is specified as a list. Example: if
            feats = [X1, X2], and lags = [[1], [1,2]], then X_1 will get lagged by one time step
            and X_2 will get lagged by two time steps. The default is None.
        local : boolean, optional
            If false, each feature sample of n_points_i will be used as input. If true, the
            surrogate will be applied locally, meaning that n_points_i separate scalar input
            features will be extracted from each feature sample. If true, the size of all
            features and target must be the same: n_points_i (for all i) = n_target = a fixed
            number. The default is False.
        test_frac : float, optional
            The final fraction of the training data that is withheld from training.
            The default is 0.0, and it must be in [0.0, 1.0].
        valid_frac : float, optional
            The fraction of the testing data that is withheld to be considered for validation.
            The default is 0.0, and it must be in [0.0, 1.0].
        train_first: boolean, if True then use first (1.0-test_frac) samples for training,
            otherwise chose training sample at random
        index: list of inidices of data samples to be chosen for training set

        Returns
        -------
        X_train:
        y_train:
        X_test:
        y_test:
        """

        if not isinstance(feats, list):
            feats = [feats]

        # the number of distinct feature arrays
        self.n_feat_arrays = len(feats)

        # flag if the surrogate is to be applied locally or not
        self.local = local

        # initialize storage for online training
        self.online_feats = [[] for i in range(self.n_feat_arrays)]
        self.online_target = []

        # number of training samples
        self.n_samples = feats[0].shape[0]
        # number of points in the computational grid
        self.n_points = feats[0].shape[1]

        # Depends on the way the test points are chosen
        # compute the size of the training set based on value of test_frac
        self.n_train = round(self.n_samples * (1.0 - test_frac))
        # number of testing points, as what is left after excluding training set
        self.n_test = np.int(self.n_samples - self.n_train)
        # get indices of samples  to be used for training
        # 1) train_first True: choose first (1-test_frac) fraction of the data set points, if points arranged in time
        # 2) train_first False: choose (1-test_frac) fraction of data set at
        # random without replacement
        if train_first:
            # chose train fraction from first sims
            self.train_indices = np.arange(self.n_samples)[:self.n_train]
            self.test_indices = np.arange(self.n_samples)[self.n_train:]
        else:
            self.train_indices = np.random.choice(self.n_samples, self.n_train, replace=False)
            self.train_indices = np.sort(self.train_indices)
            self.test_indices = np.array(
                [el for el in list(range(0, self.n_samples)) if el not in self.train_indices])

        # TODO: for GP and other models bad for extrapolation:
        #  add option to prioritize training samples at the border of presented parameter space
        print('Using  %d/%d samples to train the ML model' % (self.n_train, self.n_samples))

        if index is not None:
            self.n_train = len(index)
            self.n_test = self.n_samples - self.n_train
            self.train_indices = index
            self.test_indices = np.array(
                [el for el in list(range(0, self.n_samples)) if el not in self.train_indices])

        X = {}
        y = {}
        if self.n_test > 0:
            X_r = {}
            y_r = {}
        # use the entire row as a feature
        if not local:
            # list of features
            X[0] = [X_i[self.train_indices] for X_i in feats]  # chose train fraction randomly
            # the target data
            y[0] = target[self.train_indices]
            if self.n_test > 0:
                X_r[0] = [X_i[self.test_indices] for X_i in feats]  # chose train fraction randomly
                y_r[0] = target[self.test_indices]
        # do not use entire row as feature, apply surrogate locally along second dimension
        else:
            # create a separate training set for every grid point
            for i in range(self.n_points):
                X[i] = [X_i[self.train_indices, i] for X_i in feats]
                y[i] = target[self.train_indices].reshape([-1, 1])
                if self.n_test > 0:
                    X_r[i] = [X_i[self.test_indices, i] for X_i in feats]
                    y_r[i] = target[self.test_indices, i].reshape([-1, 1])

        X_train = []
        y_train = []

        X_test = []
        y_test = []

        # No time-lagged training data
        if lags is not None:
            self.max_lag = np.max(list(chain(*lags)))
            print('Creating time-lagged training data...')
            # lag every training set in X and y
            for i in range(len(X)):
                X_train_i, y_train_i = self.lag_training_data(X[i], y[i], lags=lags)
                X_train.append(X_train_i)
                y_train.append(y_train_i)
            if self.n_test > 0:
                # lag testing set as well, so it correspond to new features generated during lagging
                # NB: works only for train_first=True
                for i in range(len(X_r)):
                    X_test_i, y_test_i = self.lag_training_data(X_r[i], y_r[i], lags=lags)
                    X_test.append(X_test_i)
                    y_test.append(y_test_i)
        else:
            self.max_lag = 0
            # no time lag, just add every entry in X and y to an array
            # loop over all spatial points in the case of a local surrogate. For non-local
            # surrogates, len(X) = 1
            for i in range(len(X)):

                # if only one unique feature vector is used
                if len(X[i]) == 1:
                    X_train.append(np.array(X[i]).reshape([self.n_train, -1]))
                # if there are multiple feature vectors, concatenate them
                else:
                    #X_train.append(np.concatenate(X[i], axis=1))
                    X_train.append(np.moveaxis(np.array(X[i]), 0, -1).reshape([self.n_train, -1]))

                y_train.append(y[i])
                # Testing data
                if self.n_test > 0:
                    # appends same feature values in a single nfeat-long array
                    X_test.append(np.moveaxis(np.array(X_r[i]), 0, -1).reshape([self.n_test, -1]))
                    y_test.append(y_r[i])

        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        # Testing and validation data
        if self.n_test > 0:
            if valid_frac > 0.0:
                # validation fraction is extracted from original test fraction
                # (valid_frac always has t0 be lesser than test_frac)
                self.n_valid = (self.n_test + self.n_train) * valid_frac
                X_valid = X_test[-self.n_valid:]
                X_test = X_test[:-self.n_valid]
                y_valid = y_test[-self.n_valid:]
                y_test = y_test[:-self.n_valid]
                X_valid = np.concatenate(X_valid)
                y_valid = np.concatenate(y_valid)

            X_test = np.concatenate(X_test)
            y_test = np.concatenate(y_test)
            print('done preparing data')
        else:
            X_test = np.empty((0, X_train.shape[1]))
            y_test = np.empty((0, y_train.shape[1]))

        return X_train, y_train, X_test, y_test

    def get_online_training_data(self, **kwargs):
        """
        Return the training data for a single online-learning step.

        Returns
        -------
        X_train : array
            The feature array.
        y_train : array
            The target array.

        """

        feats = self.online_feats
        target = np.array(self.online_target)

        # if lagged feature vectors are used
        if self.lags is not None:

            feats = [np.array(feat) for feat in feats]
            if not self.local:
                # create (time-lagged) training data from X
                X_train, y_train = self.lag_training_data(feats, target, lags=self.lags,
                                                          init_feats=False)
            else:
                X_train = []
                y_train = []
                # create a separate training set for every grid point
                for i in range(self.n_points):
                    X_i = [X_i[:, i] for X_i in feats]
                    y_i = target[:, i].reshape([-1, 1])
                    # create time-lagged data per gridpoint
                    X_train_i, y_train_i = self.lag_training_data(X_i, y_i, lags=self.lags,
                                                                  init_feats=False)
                    X_train.append(X_train_i)
                    y_train.append(y_train_i)
                X_train = np.concatenate(X_train)
                y_train = np.concatenate(y_train)

        # do not time lag data
        else:
            # make a single array where each row contains a concateated vector of all feature
            # vectors
            X_train = np.concatenate(feats, axis=1)
            X_train = X_train.reshape([-1, kwargs['n_in']])
            y_train = target.reshape([-1, kwargs['n_out']])

        return X_train, y_train

    def generate_online_training_data(self, feats, LR_before, LR_after, HR_before, HR_after):
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
        LR_before : array
            Low resolution state at previous time step.
        LR_after : array
            Low resolution state at current time step.
        HR_before : array
            High resolution state at previous time step.
        HR_after : array
            High resolution state at current time step.

        Returns
        -------
        None.

        """

        # multiple faetures arrays are stored in a list. For consistency put a single
        # array also in a list.
        if isinstance(feats, np.ndarray):
            feats = [feats]

        # store input features
        for i in range(self.n_feat_arrays):
            self.online_feats[i].append(feats[i])

        # difference of the low res model between time n and time n+1
        delta_LR = LR_after - LR_before
        # the difference between the low res and high res model at time n
        delta_nudge = LR_before - HR_before
        # difference of the high res model between time n and time n+1
        delta_HR = HR_after - HR_before

        # assume that over a small time interval [n, n+1] we can write:
        # delta_HR = would_be_delta_without_nuding + delta_due_to_nudging
        delta_no_nudge_HR = delta_HR - delta_nudge / self.tau_nudge * self.dt_LR

        # compute correction from: delta_LR + correction = delta_no_nudge_HR
        correction = delta_no_nudge_HR - delta_LR

        # make the correction the target for the neural network. Divide by timestep
        # since update is LR += correction * dt
        self.online_target.append(correction / self.dt_LR)

        # remove oldest item from the online features is the window lendth is exceeded
        if len(self.online_feats[0]) > self.window_length:
            for i in range(self.n_feat_arrays):
                self.online_feats[i].pop(0)
            self.online_target.pop(0)

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
        self.tau_nudge = tau_nudge
        self.dt_LR = dt_LR
        self.window_length = window_length

    def lag_training_data(self, X, y, lags, init_feats=True):
        """
        Create time-lagged supervised training data X, y

        Parameters:
            X: features. Either an array of dimension (n_samples, n_features)
               or a list of arrays of dimension (n_samples, n_features)

            y: training target. Array of dimension (n_samples, n_outputs)

            lags: list of lists, containing the integer values of lags
                  Example: if X=[X_1, X_2] and lags = [[1], [1, 2]], the first
                  feature array X_1 is lagged by 1 (time) step and the second
                  by 1 and 2 (time) steps.

        Returns:
            X_train, y_trains (arrays), of lagged features and target data. Every
            row of X_train is one (time) lagged feature vector. Every row of y_train
            is a target vector at the next (time) step
        """

        # compute the max number of lags in lags
        lags_flattened = list(chain(*lags))
        max_lag = np.max(lags_flattened)

        # total number of data samples
        n_samples = y.shape[0]

        # if X is one array, add it to a list anyway
        if isinstance(X, np.ndarray):
            tmp = []
            tmp.append(X)
            X = tmp

        # compute target data at next (time) step
        if y.ndim == 2:
            y_train = y[max_lag:, :]
        elif y.ndim == 1:
            y_train = y[max_lag:]
        else:
            print("Error: y must be of dimension (n_samples, ) or (n_samples, n_outputs)")
            return

        # a lag list must be specified for every feature in X
        if len(lags) != len(X):
            print('Error: no specified lags for one of the featutes in X')
            return

        # compute the lagged features
        C = []
        idx = 0
        for X_i in X:

            for lag in np.sort(lags[idx])[::-1]:
                begin = max_lag - lag
                end = n_samples - lag

                if X_i.ndim == 2:
                    C.append(X_i[begin:end, :])
                elif X_i.ndim == 1:
                    C.append(X_i[begin:end])
                else:
                    print("Error: X must contains features of dimension (n_samples, ) \
                          or (n_samples, n_features)")
                    return
            idx += 1

        # C is a list of lagged features, turn into a single array X_train
        X_train = C[0]

        if X_train.ndim == 1:
            X_train = X_train.reshape([y_train.shape[0], 1])

        for X_i in C[1:]:

            if X_i.ndim == 1:
                X_i = X_i.reshape([y_train.shape[0], 1])

            X_train = np.append(X_train, X_i, axis=1)

        # initialize the storage of features
        if init_feats:
            self.empty_feature_history(lags)

        return X_train, y_train

    def bin_data(self, y, n_bins):
        """
        Bin the data y in to n_bins non-overlapping bins

        Parameters
        ----------
        y:  array
            size (number of samples, number of variables): Data
        n_bins: int
             Number of (equidistant) bins to be used.

        Returns
        -------
        None.

        """

        n_samples = y.shape[0]

        if y.ndim == 2:
            n_vars = y.shape[1]
        else:
            n_vars = 1
            y = y.reshape([n_samples, 1])

        self.binnumbers = np.zeros([n_samples, n_vars]).astype('int')
        self.y_binned = {}
        self.y_binned_mean = {}
        y_idx_binned = np.zeros([n_samples, n_bins * n_vars])
        self.bins = {}
        self.n_vars = n_vars

        for i in range(n_vars):

            self.y_binned[i] = {}
            self.y_binned_mean[i] = {}

            bins = np.linspace(np.min(y[:, i]), np.max(y[:, i]), n_bins + 1)
            self.bins[i] = bins

            _, _, self.binnumbers[:, i] = \
                stats.binned_statistic(y[:, i], np.zeros(n_samples), statistic='count', bins=bins)

            unique_binnumbers = np.unique(self.binnumbers[:, i])

            offset = i * n_bins

            for j in unique_binnumbers:
                idx = np.where(self.binnumbers[:, i] == j)
                self.y_binned[i][j - 1] = y[idx, i]
                self.y_binned_mean[i][j - 1] = np.mean(y[idx, i])
                y_idx_binned[idx, offset + j - 1] = 1.0

        return y_idx_binned

    def empty_feature_history(self, lags):
        """
        Initialize an empty feat_history dict. This dict keeps track of the features
        arrays that were used up until 'max_lag + 1' steps ago.

        Parameters:

            lags: list of lists, containing the integer values of lags
                  Example: if X=[X_1, X_2] and lags = [[1], [1, 2]], the first
                  feature array X_1 is lagged by 1 (time) step and the second
                  by 1 and 2 (time) steps.
        """
        self.lags = []

        for l in lags:
            self.lags.append(np.sort(l)[::-1])

        # self.max_lag = np.max(list(chain(*lags)))

        self.feat_history = {}

        # the number of feature arrays that make up the total input feature vector
        # self.n_feat_arrays = len(lags)

        for i in range(self.n_feat_arrays):
            self.feat_history[i] = []

    def initial_condition_feature_history(self, feats, start=0):
        """
        The features can be lagged in time. Therefore, the initial condition of the
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
            if not self.local:
                feat = [X_i[start + i] for X_i in feats]
            else:
                feat = [X_i[start + i].reshape([1, -1]) for X_i in feats]

            self.append_feat(feat)

    def append_feat(self, X):
        """
        Append the feature vectors in X to feat_history dict

        Parameters:

            X: features. Either an array of dimension (n_samples, n_features)
               or a list of arrays of dimension (n_samples, n_features)

        """

        # if X is one array, add it to a list anyway
        if isinstance(X, np.ndarray):
            X = [X]

        for i in range(self.n_feat_arrays):

            assert isinstance(
                X[i], np.ndarray), 'ERROR: Only numpy arrays are allowed as input features.'

            self.feat_history[i].append(X[i])

            # if max number of features is reached, remove first item
            if len(self.feat_history[i]) > self.max_lag + 1:
                self.feat_history[i].pop(0)

    def get_feat_history(self, **kwargs):
        """
        Return the features from the feat_history dict based on the lags
        specified in self.lags

        Returns:
            X_i: array of lagged features of dimension (feat1.size + feat2.size + ...,)
        """
        X_i = []

        idx = 0
        for i in range(self.n_feat_arrays):
            for lag in self.lags[idx]:
                begin = self.max_lag - lag
                current_feat = self.feat_history[i][begin]
                if current_feat.ndim == 1:
                    X_i.append(current_feat)
                elif current_feat.shape[1] == 1:
                    X_i.append(current_feat.flatten())
                else:
                    X_i.append(np.array([current_feat[0][kwargs['index']]]))
            idx += 1

        return np.array(list(chain(*X_i)))

    # def standardize_data(self, standardize_X=True, standardize_y=True):
    #     """
    #     Standardize the training data
    #     """

    #     if standardize_X:
    #         X_mean = np.mean(self.X, axis=0)
    #         X_std = np.std(self.X, axis=0)
    #     else:
    #         X_mean = 0.0
    #         X_std = 1.0

    #     if standardize_y:
    #         y_mean = np.mean(self.y, axis=0)
    #         y_std = np.std(self.y, axis=0)
    #     else:
    #         y_mean = 0.0
    #         y_std = 1.0

    #     return (self.X - X_mean) / X_std, (self.y - y_mean) / y_std

    # def recursive_moments(self, X_np1, mu_n, sigma2_n, N):
    #     """
    #     Recursive formulas for the mean and variance. Computes the new moments
    #     when given a new sample and the old moments.

    #     Arguments
    #     ---------
    #     + X_np1: a new sample of random variable X
    #     + mu_n: the sample mean of X, not including X_np1
    #     + sigma2_n: the variance of X, not including X_np1
    #     + N: the total number of samples thus far

    #     Returns
    #     -------
    #     The mean and variance, updated based on the new sample

    #     """
    #     mu_np1 = mu_n + (X_np1 - mu_n) / (N + 1)
    #     sigma2_np1 = sigma2_n + mu_n**2 - mu_np1**2 + (X_np1**2 - sigma2_n - mu_n**2) / (N + 1)

    #     return mu_np1, sigma2_np1

    # def estimate_embedding_dimension(self, y, N):

    #     for n in range(3, N+1):

    #         lags_n = range(1, n)
    #         lags_np1 = range(1, n+1)
    #         y_n, _ = self.lag_training_data(y, np.zeros(y.size), [lags_n])
    #         y_np1, _ = self.lag_training_data(y, np.zeros(y.size), [lags_np1])

    #         L = y_np1.shape[0]
    #         dist_n = np.zeros(L)
    #         dist_np1 = np.zeros(L)

    #         for l in range(L):
    #             # d = np.linalg.norm(y_n - y_n[l], axis = 0)
    #             d = np.sum((y_n - y_n[l])**2, axis=1)
    #             a, d_min = np.partition(d, 1)[0:2]
    #             # d = np.linalg.norm(y_np1 - y_np1[l], axis = 0)
    #             d = np.sum((y_np1 - y_np1[l])**2, axis=1)
    #             _, d_min2 = np.partition(d, 1)[0:2]

    #             dist_n[l] = d_min
    #             dist_np1[l] = d_min2

    #         test = np.abs((dist_n - dist_np1)/dist_n)

    #         print(len(lags_n))
    #         print(np.where(test > 20)[0].size)
