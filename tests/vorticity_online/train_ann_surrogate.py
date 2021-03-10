from itertools import chain
import numpy as np
import easysurrogate as es
import matplotlib.pyplot as plt

# create EasySurrogate campaign
campaign = es.Campaign()

# load HDF5 data frame
data_frame = campaign.load_hdf5_data()

# supervised training data set
Q_LR = data_frame['Q_LR']
c_ij = data_frame['c_ij']
c_ij = c_ij.reshape([c_ij.shape[0], c_ij.shape[1]])
src_Q = data_frame['src_Q']

features = [Q_LR, c_ij, src_Q]
target = data_frame['Q_HR'] - data_frame['Q_LR']

# create a (time-lagged) ANN surrogate
surrogate = es.methods.ANN_Surrogate()

# create time-lagged features
lags = [range(1, 20), range(1, 20), range(1, 20)]
# lags = None

# train the surrogate on the data
n_iter = 100000
local = False
surrogate.train(features, target, n_iter, lags=lags, n_layers=4, n_neurons=50,
                batch_size=512, test_frac=0.5)

campaign.add_app(name='test_campaign', surrogate=surrogate)
campaign.save_state(file_path='../samples/campaign.pickle')

# plot fit
fig = plt.figure(figsize=[4, 4])
ax = fig.add_subplot(111)
# I = features[0].shape[0] - surrogate.max_lag
I = surrogate.neural_net.n_train
ax.plot(target[0:I, 0], 'b.')
prediction = np.zeros([I])
for i in range(I):
    r = surrogate.predict([features[k][surrogate.max_lag + i] for k in range(len(features))])
    prediction[i] = r[0]

ax.plot(prediction, 'y.')
plt.tight_layout()
plt.show()
