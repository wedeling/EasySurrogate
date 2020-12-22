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
lags = [range(11)]
lags = None
local = False

# create the KDE anchor points and standard deviations
n_means = 15
n_stds = 3
kernel_means = []
kernel_stds = []

if not local:
    n_softmax = target.shape[1]
    for i in range(n_softmax):
        kernel_means.append(np.linspace(np.min(target[:, i]), np.max(target[:, i]), n_means))
        kernel_stds.append(np.linspace(0.2, 0.3, n_stds))
    kernel_means = np.array(kernel_means)
    kernel_stds = np.array(kernel_stds)
else:
    n_softmax = 1
    kernel_means.append(np.linspace(np.min(target), np.max(target), n_means))
    kernel_stds.append(np.linspace(0.2, 0.3, n_stds))
    kernel_means = np.array(kernel_means)
    kernel_stds = np.array(kernel_stds)

# train the surrogate on the data
n_iter = 5000
surrogate.train([features], target, n_iter,
                kernel_means, kernel_stds, n_softmax,
                lags=lags, local=local,
                n_layers=3, n_neurons=50,
                batch_size=512, test_frac=0.5)

campaign.add_app(name='test_campaign', surrogate=surrogate)
campaign.save_state()

# KMN analysis object
analysis = es.analysis.KMN_analysis(campaign.surrogate)
analysis.make_movie()
