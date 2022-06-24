import easysurrogate as es
import sys
import json

# read the current hyperparameter values
json_input = sys.argv[1]
with open(json_input, "r") as f:
    inputs = json.load(f)

# create EasySurrogate campaign
campaign = es.Campaign()

# load HDF5 data frame
data_frame = campaign.load_hdf5_data()

# supervised training data set
features = data_frame['X_data']
target = data_frame['B_data']

# create a ANN surrogate
surrogate = es.methods.ANN_Surrogate()

# create time-lagged features
lags = [[1, 10]]

# train the surrogate on the data
n_iter = 1000

surrogate.train([features], 
                target, 
                n_iter, 
                lags=lags, 
                n_layers=inputs['n_layers'], 
                n_neurons=inputs['n_neurons'],
                batch_size=512, 
                test_frac=0.2)

campaign.add_app(name='test_campaign', surrogate=surrogate)
campaign.save_state()