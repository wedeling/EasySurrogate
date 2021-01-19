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
c_ij_u = data_frame['c_ij_u']
c_ij_u = c_ij_u.reshape([c_ij_u.shape[0], c_ij_u.shape[1]])
c_ij_v = data_frame['c_ij_v']
c_ij_v = c_ij_v.reshape([c_ij_v.shape[0], c_ij_v.shape[1]])
src_Q_u = data_frame['src_Q_u']
src_Q_v = data_frame['src_Q_v']

# features = [Q_LR, c_ij_u, c_ij_v, src_Q_u, src_Q_v]
features = [Q_LR]
target = data_frame['Q_HR'] - data_frame['Q_LR']

# create a (time-lagged) ANN surrogate
surrogate = es.methods.ANN_Surrogate()

# create time-lagged features
# lags = [range(1, 2)]
lags = None

# train the surrogate on the data
n_iter = 20000
local = False
surrogate.train(features, target, n_iter, lags=lags, n_layers=4, n_neurons=50,
                batch_size=512, local=local)

campaign.add_app(name='test_campaign', surrogate=surrogate)
campaign.save_state(file_path='../samples/campaign.pickle')

# plot fit
fig = plt.figure(figsize=[4, 4])
ax = fig.add_subplot(111)
ax.plot(target[:, 0], 'b.')
I = features[0].shape[0] - surrogate.max_lag
prediction = np.zeros([I])
for i in range(I):
    r = surrogate.predict([features[k][surrogate.max_lag + i] for k in range(len(features))])
    prediction[i] = r[0]

ax.plot(prediction, 'y.')
plt.tight_layout()
plt.show()
