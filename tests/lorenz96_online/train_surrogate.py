# # Note that we are technically not using this in Coupled learning
# class NNParam():
#     """Parameterization class that can be used inside a L96 object"""
#     def __init__(self, keras_net):
#         self.net = keras_net
#     def __call__(self, x):
#         return self.net.predict_on_batch(x).squeeze(-1)

from itertools import chain
import numpy as np
import easysurrogate as es
import matplotlib.pyplot as plt

# create EasySurrogate campaign
campaign = es.Campaign()

# load HDF5 data frame
data_frame = campaign.load_hdf5_data()

# number of HR time steps per LR time step. This must be used to subsample the reference data
# if a time lagged surrogate is used
N = 10

# supervised training data set
features = data_frame['X_data'][0:-1:N, :]
target = data_frame['B_data'][0:-1:N, :]

# create a (time-lagged) ANN surrogate
surrogate = es.methods.ANN_Surrogate()

# create time-lagged features
lags = [range(10)]
# lags = None

# train the surrogate on the data
n_iter = 10000
local = True
surrogate.train([features], target, n_iter, lags=lags, n_layers=3, n_neurons=50,
                batch_size=512, local=local)

campaign.add_app(name='test_campaign', surrogate=surrogate)
campaign.save_state(file_path='../samples/campaign.pickle')

# plot fit
fig = plt.figure(figsize=[4, 4])
ax = fig.add_subplot(111)

ax.plot(features, target, 'b.', alpha=0.2)

I = 2000
start = surrogate.max_lag

for i in range(I):
    if local:
        B = surrogate.predict(features[start + i].reshape([1, 18]))
    else:
        B = surrogate.predict(features[start + i])

    ax.plot(features[start + i], B, 'y.', alpha=0.15)
plt.tight_layout()
plt.show()

####################

# from tensorflow import keras
# from tensorflow.keras.layers import *

# wrong_nn = keras.models.Sequential([
#     Dense(32, input_shape=(1,), activation='elu'),
#     Dense(32, activation='elu'),
#     Dense(1)
# ])

# wrong_nn.compile('adam', 'mse')

# # Should take around half a minute
# wrong_nn.fit(features, target, epochs=30, verbose=1, batch_size=1024)

# # Save the neural network parameters so we can load them again
# wrong_nn.save('./nn.h5')

# #####################

# # Load the "wrongly" pretrained model
# nn = keras.models.load_model('./nn.h5')
# ml_param = NNParam(nn)

# #plot fit
# fig = plt.figure(figsize=[4,4])
# ax = fig.add_subplot(111)

# ax.plot(features[::200], target[::200], 'b.', alpha=0.2)

# N = 100
# a = np.linspace(-5, 10, N)
# fit = ml_param(a)

# ax.plot(a, fit, 'y')
# plt.tight_layout()
# plt.show()
