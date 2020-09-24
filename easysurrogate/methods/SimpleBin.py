import numpy as np
from scipy import stats
from itertools import chain


class SimpleBin:

    def __init__(self, feat_eng):
        """
        Parameters
        ----------
        feat_eng : EasySurrogate Feature_Engineering object

        Returns
        -------
        None.

        """

        if not hasattr(feat_eng, 'y_binned'):
            print("Error: feat_eng object does not contain binned data")
            print("Run the bin_data(..) subroutine of feat_eng")
            return

        # number of variables
        self.n_vars = feat_eng.n_vars

        # dict of the binned data. To access the binned data of bin 1 of the
        # 6-th variable use: y_binned[5][1]
        self.y_binned = feat_eng.y_binned
        self.y_binned_mean = feat_eng.y_binned_mean

    def resample(self, bin_idx):
        """
        Resamples reference data from bins specified by bin indices bin_idx.
        Bin indices are integers >= 1.

        Parameters
        ----------
        bin_idx : array of integers, size (nvars,): the bin indices of each
                  output variable

        Returns
        -------
        pred : array of floats, size (nvars,): array of resampled reference
               data. Samples are drawn from the bins specified by bin_idx

        """

        pred = np.zeros(self.n_vars)

        for i in range(self.n_vars):
            pred[i] = np.random.choice(self.y_binned[i][bin_idx[i]][0])

        return pred

    def resample_mean(self, bin_idx):
        """
        Resamples reference bin mean specified by bin indices bin_idx.
        Bin indices are integers >= 1.

        Parameters
        ----------
        bin_idx : array of integers, size (nvars,): the bin indices of each
                  output variable

        Returns
        -------
        pred : array of floats, size (nvars,): array of resampled bin mean
               data.

        """

        pred = np.zeros(self.n_vars)

        for i in range(self.n_vars):
            pred[i] = self.y_binned_mean[i][bin_idx[i]]

        return pred
