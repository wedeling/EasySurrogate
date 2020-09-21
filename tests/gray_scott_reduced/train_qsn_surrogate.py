from itertools import chain
import numpy as np
import easysurrogate as es

# create EasySurrogate campaign
load_state = True
campaign = es.Campaign(load_state=load_state)

# load HDF5 data frame
data_frame = campaign.load_hdf5_data()

# supervised training data set
c_ij_u = data_frame['c_ij_u']
c_ij_u = c_ij_u.reshape([c_ij_u.shape[0], c_ij_u.shape[1]])
features = [c_ij_u, data_frame['src_Q_u']]
targets = data_frame['tau_u']

if not load_state:
    # create Quantized Softmax Network surrogate
    surrogate = es.methods.QSN_Surrogate()
    
    # # create time-lagged features
    lags = [range(1,75), range(1,75)]
    
    # train the surrogate on the data
    n_iter = 40000
    surrogate.train(features, targets, lags, n_iter, n_layers=4, n_neurons=256,
                    batch_size=500, activation='hard_tanh')
    
    campaign.add_app(name='test_campaign', surrogate=surrogate)
    campaign.save_state()
else:
    surrogate = campaign.surrogate

# QSN analysis object
analysis = es.analysis.QSN_analysis(surrogate)
analysis.get_classification_error(features, targets)
