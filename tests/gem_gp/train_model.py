import time as t
import numpy as np

import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux', 'te_transp_flux_std', 'ti_transp_flux_std']

SEQDES = False #True

campaign = es.Campaign(load_state=False)

# 1) Case for data from single flux tube GEM UQ campaign
#data_file_name = 'gem_data_625.hdf5'
#features_names_selected = features_names
#target_name_selected = target_names

# 2) Case for data from a MFW production run
#data_file_name = 'gem_workflow_500.hdf5'
#features_names_selected = features_names
#target_name_selected = target_names

# 3) Case for data generated from single flux tube GEM0 with 4 parameters (LHD, with a wrapper)
#data_file_name = 'gem0_lhc.hdf5'
#features_names_selected = features_names
#target_name_selected = target_names

# 4) Case for from single flux tube GEM0 with 2 parameters (LHD, with a wrapper)
#data_file_name = 'gem0_lhc_256.hdf5'
#features_names_selected = [features_names[2], features_names[3]]
#target_name_selected = [target_names[1]]

# 5) Case from single flux tube GEM UQ campaign (4 parameters, tensor product of grid with 2 points per DoF)
data_file_name = 'gem_uq_16_std.hdf5'
features_names_selected = features_names
target_name_selected = [target_names[3]]

# Create a surrogate and its model; train and save it

# Create a campaign object
data_frame = campaign.load_hdf5_data(file_path=data_file_name)

# prepare lists of features and array of targets
features = [data_frame[k] for k in features_names_selected if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_name_selected if k in data_frame], axis=1)

time_init_start = t.time()

# TODO: form a surrogate model parameter dictionary and pass starting from here  
gp_param = {
            'bias': True,
            'nonstationary': True,
           }

surrogate = es.methods.GP_Surrogate(
                            backend='local',
                            n_in=len(features),
                                   )

print('Time to initialise the surrogate: {:.3} s'.format(t.time() - time_init_start))

time_train_start = t.time()

# TODO pass hyperparameters here!
surrogate.train(features, target, 
                test_frac=0.5,
                bias=gp_param['bias'],
                nonstationary=gp_param['nonstationary']
               )

print('Time to train the surrogate: {:.3} s'.format(t.time() - time_train_start))
surrogate.model.print_model_info()

save_model_file_name = 'model_STD_05train_10062022.pickle'

campaign.add_app(name='gp_campaign', surrogate=surrogate)
campaign.save_state(file_path=save_model_file_name)

# Sequential Design Modification
if SEQDES:
    surrogate.train_sequentially(features, target, n_iter=10, save_history=True)
    surrogate.model.print_model_info()
    campaign.add_app(name='gp_campaign_sequential', surrogate=surrogate)
    campaign.save_state()

