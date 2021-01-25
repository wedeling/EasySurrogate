import timeit
import numpy as np
import csv

import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux']

# Create a campaign object
campaign = es.Campaign(load_state=False)

data_frame = campaign.load_hdf5_data(file_path='gem_data_625.hdf5')

#features = [np.array(data_frame[k]).transpose().reshape((-1, 1)) for k in features_names if k in data_frame]
#target = np.array([data_frame[k] for k in target_names if k in data_frame]).transpose()

features = [data_frame[k] for k in features_names if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_names if k in data_frame], axis=1)

# Create a surrogate and its app
surrogate = es.methods.GP_Surrogate()
surrogate.train(features, target, test_frac=0.5)
#surrogate.train(features, target, test_frac=0.0, postrain=True)  # try to retrain on max uncertain samples of full feature dataset
campaign.add_app(name='gp_campaign', surrogate=surrogate)
campaign.save_state()
