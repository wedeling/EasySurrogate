import time as t
import numpy as np
import sys
from datetime import datetime

import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux', 'te_transp_flux_std', 'ti_transp_flux_std']

rs = 42
np.random.seed(rs)

SEQDES = False

code_name = 'gem0'

if len(sys.argv) < 2 :
    index = 0
else:  
    index = sys.argv[1]

if len(sys.argv) < 3 :
    date_gen = "20231216"
else:  
    date_gen = sys.argv[2]

date_data = "20240110"

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
# #data_file_name = 'gem_uq_16_std.hdf5'
# #data_file_name = 'gem_uq_79_std.hdf5'
# data_file_name = 'gem_uq_81_std.hdf5'
# features_names_selected = features_names
# target_name_selected = [target_names[1]] # model for [1 - means ; 3- std] of data

# ... 
# # 8) Case from 8 flux tube GEM0 5000 runs (4 parameters, tensor product of grid with 5 points per DoF)

# data_file_name = f"{code_name}_5000_transp_{index}_20231216.hdf5"

# features_names_selected = features_names
# target_name_selected = [target_names[0], target_names[1]]

# 9) Case from 8 flux tube GEM0 8000 runs (4 parameters, 8 flux tubes, 10**3 LHC samples per flux tube)

n_samples = 8000

data_file_name = f"{code_name}_{n_samples}_transp_{index}_{date_data}.hdf5"

features_names_selected = features_names
target_name_selected = [target_names[0], target_names[1]]

###
# Create a surrogate and its model; train and save it

# Create a campaign object
data_frame = campaign.load_hdf5_data(file_path=data_file_name)

# prepare lists of features and array of targets
features = [data_frame[k] for k in features_names_selected if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_name_selected if k in data_frame], axis=1)

time_init_start = t.time()

ann_param = {
            'backend': 'local', #'scikit-learn'
            #'process_type': 'student_t', #'gaussian'
            #'kernel': 'Matern', #'RBF'
            #'length_scale': 1.0,  #[1.]*len(features),
            #'noize': 0.1,
            #'bias': 0.,
            #'nonstationary': False,
            'test_frac': 0.2,
            'n_iter': 30000,
            'n_layers': 5,
            'n_neurons': 128,
            'batch_size': 32,
            'activation': 'relu', #'sigmoid', #sigmoid usually gives better RMSE but might be a reason for misfit around constant dependences
           }

surrogate = es.methods.ANN_Surrogate(
                            #backend=gp_param['backend'],
                            #n_in=len(features),
                                   )

print('Time to initialise the surrogate: {:.3} s'.format(t.time() - time_init_start))

time_train_start = t.time()

surrogate.train(features, 
                target, 
                test_frac=ann_param['test_frac'],
                n_iter=ann_param['n_iter'],
                n_layers=ann_param['n_layers'],
                n_neurons=ann_param['n_neurons'],
                batch_size=ann_param['batch_size'],
                activation=ann_param['activation'],
               )

print('Time to train the surrogate: {:.3} s'.format(t.time() - time_train_start))
surrogate.neural_net.print_network_info()

date_str = datetime.now().strftime("%Y%m%d")
save_model_file_name = f"model_{code_name}_{n_samples}_tf{ann_param['test_frac']}_{len(features_names_selected)}i{len(target_name_selected)}o_ft{index}_ann{ann_param['n_layers']}x{ann_param['n_neurons']}x{ann_param['batch_size']}_{date_str}.pickle"

campaign.add_app(name='ann_campaign', surrogate=surrogate)
campaign.save_state(file_path=save_model_file_name)

# # Sequential Design Modification
# if SEQDES:
#     surrogate.train_sequentially(features, target, n_iter=10, save_history=True)
#     surrogate.model.print_model_info()
#     campaign.add_app(name='gp_campaign_sequential', surrogate=surrogate)
#     campaign.save_state()
