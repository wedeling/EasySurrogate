from itertools import chain
import numpy as np
import easysurrogate as es

# create EasySurrogate campaign
campaign = es.Campaign()

# load HDF5 data frame
data_frame = campaign.load_hdf5_data()

# supervised training data set
features = data_frame['X_data']
target = data_frame['B_data']

# create (time-lagged) ANN surrogate
surrogate = es.methods.ANN_Surrogate()

# create time-lagged features
lags = [[1, 10]]

# train the surrogate on the data
n_iter = 10000
surrogate.train([features], target, lags, n_iter, n_layers=4, n_neurons=256,
                batch_size=512)

campaign.add_app(name='test_campaign', surrogate=surrogate)
campaign.save_state()