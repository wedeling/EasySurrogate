"""
Here a pointwise surrogate is trained, with feature vectors that include the local point
plus a number of neighbouring points.
"""

from itertools import chain
import numpy as np
import easysurrogate as es

# create EasySurrogate campaign
campaign = es.Campaign()

# load HDF5 data frame
data_frame = campaign.load_hdf5_data()

# supervised training data set
I = 9
features = data_frame['X_data'][:, I-1:I+2]
target = data_frame['B_data'][:, I].reshape([-1,1])

# create Quantized Softmax Network surrogate
surrogate = es.methods.QSN_Surrogate()

# create time-lagged features
lags = [range(1, 10)]

# train the surrogate on the data
n_iter = 10000
surrogate.train([features], target, n_iter, lags=lags, n_layers=4, n_neurons=256,
                batch_size=512)

campaign.add_app(name='test_campaign', surrogate=surrogate)
campaign.save_state()

# QSN analysis object
analysis = es.analysis.QSN_analysis(surrogate)
analysis.get_classification_error(index=np.arange(0, 10000))
