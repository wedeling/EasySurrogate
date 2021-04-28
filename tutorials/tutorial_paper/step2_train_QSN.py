from itertools import chain
import numpy as np
import easysurrogate as es

# Create EasySurrogate campaign
campaign = es.Campaign()

# Load HDF5 data frame
data_frame = campaign.load_hdf5_data()

# Supervised training data set
features = data_frame['X_n']
target = data_frame['r_n']

# create Quantized Softmax Network surrogate
surrogate = es.methods.QSN_Surrogate()

# create time-lagged features
lags = [range(1, 75)]

# train the surrogate on the data
n_iter = 20000
surrogate.train(features, target, n_iter, lags=lags, n_layers=4, n_neurons=256,
                batch_size=512, test_frac=0.5)

campaign.add_app(name='L96_campaign', surrogate=surrogate)
campaign.save_state()

# QSN analysis object
analysis = es.analysis.QSN_analysis(surrogate)
analysis.get_classification_error(index=np.arange(0, 10000))