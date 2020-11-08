from itertools import chain
import numpy as np
import easysurrogate as es
import matplotlib.pyplot as plt

# create EasySurrogate campaign
campaign = es.Campaign()

# load HDF5 data frame
data_frame = campaign.load_hdf5_data()

# supervised training data set
features = data_frame['X_data'].flatten()
target = data_frame['B_data'].reshape([-1, 1])

# create a (time-lagged) ANN surrogate
surrogate = es.methods.ANN_Surrogate()

# create time-lagged features
lags = [[1]]

# train the surrogate on the data
n_iter = 10000
surrogate.train([features], target, lags, n_iter, n_layers=3, n_neurons=32,
                batch_size=512)

campaign.add_app(name='test_campaign', surrogate=surrogate)
campaign.save_state()

#plot fit
fig = plt.figure(figsize=[4,4])
ax = fig.add_subplot(111)

ax.plot(features[::200], target[::200], 'b.', alpha=0.2)

N = 100
a = np.linspace(-5, 10, N)
fit = []
for i in range(N):
    fit.append(surrogate.predict(a[i]))
    
ax.plot(a, fit, 'y')
plt.tight_layout()
plt.show()