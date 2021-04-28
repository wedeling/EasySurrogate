import time as t
import numpy as np

import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux']

#features_names_selected = [features_names[1], features_names[3]]
features_names_selected = features_names
target_name_selected = [target_names[1]]

data_file_name = 'gem_data_625.hdf5'
#data_file_name = 'gem_workflow_500.hdf5'
#data_file_name = 'gem0_lhc.hdf5'

# Create a campaign object
campaign = es.Campaign(load_state=False)
data_frame = campaign.load_hdf5_data(file_path=data_file_name)

#features = [np.array(data_frame[k]).transpose().reshape((-1, 1)) for k in features_names if k in data_frame]
#target = np.array([data_frame[k] for k in target_names if k in data_frame]).transpose()

features = [data_frame[k] for k in features_names_selected if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_name_selected if k in data_frame], axis=1)

# Create a surrogate and its app
time_init_start = t.time()
surrogate = es.methods.GP_Surrogate()
print('Time to initialise the surrogate: {:.3} s'.format(t.time() - time_init_start))

time_train_start = t.time()
surrogate.train(features, target, test_frac=0.0)
print('Time to train the surrogate: {:.3} s'.format(t.time() - time_train_start))
surrogate.model.print_model_info()  # TODO the result of fitting does not change the default hyperparameter values


campaign.add_app(name='gp_campaign', surrogate=surrogate)
campaign.save_state()
