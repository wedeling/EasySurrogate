import easysurrogate as es
import numpy as np

campaign = es.Campaign()

data_frame = campaign.load_data()
features = data_frame['X_data']
target = data_frame['B_data']

surrogate = es.methods.QSN_Surrogate()

lags = [range(1, 75)]
n_iter = 10000
surrogate.train(features, target, lags, n_iter, n_layers=4, n_neurons=256)