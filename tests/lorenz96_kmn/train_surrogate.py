from itertools import chain
import numpy as np
import easysurrogate as es

# create EasySurrogate campaign
campaign = es.Campaign(load_state=False)

# load HDF5 data frame
data_frame = campaign.load_hdf5_data()

# supervised training data set
features = data_frame['X_data']
target = data_frame['B_data']

# create Kernel Mixture Network surrogate
surrogate = es.methods.KMN_Surrogate()

# create time-lagged features
lags = [[1, 10]]

#create the KDE anchor points and standard deviations
#NOTE: could proba move this inside the surrogate class? although stds are hard to estmate
#would have t use a standard bandwidth estimator
n_means = 15; n_stds = 3
n_out = target.shape[1]
kernel_means = np.zeros([n_out, n_means])
kernel_stds = np.zeros([n_out, n_stds])
for i in range(n_out):
    kernel_means[i] = np.linspace(np.min(target[:, i]), np.max(target[:, i]), n_means)
    kernel_stds[i] = np.linspace(0.2, 0.3, n_stds)

# train the surrogate on the data
n_iter = 10000
surrogate.train([features], target, lags, n_iter, 
                kernel_means, kernel_stds,
                n_layers=4, n_neurons=256,
                batch_size=512)

campaign.add_app(name='test_campaign', surrogate=surrogate)
campaign.save_state()

# KMN analysis object
analysis = es.analysis.KMN_analysis(campaign.surrogate)
analysis.make_movie()