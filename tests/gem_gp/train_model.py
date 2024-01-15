import time as t
import numpy as np
import sys
from datetime import datetime

import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux', 'te_transp_flux_std', 'ti_transp_flux_std']

if len(sys.argv) < 2 :
    index = 0
else:
    index = sys.argv[1]

if len(sys.argv) < 3 :
    date_gen = "20231216"
else:  
    date_gen = sys.argv[2]

code_name = 'gem0'

np.random.seed(42)

SEQDES = False #True if Sequential Design of Experiments to be used

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
#data_file_name = 'gem_uq_16_std.hdf5'
#data_file_name = 'gem_uq_79_std.hdf5'
# data_file_name = 'gem_uq_81_std.hdf5'
# features_names_selected = features_names
# target_name_selected = [target_names[1]] # model for [1 - means ; 3- std] of data

# 6) Case from single flux tube GEM UQ campaign (4 parameters, tensor product of grid with 2 points per DoF)
#                       and 2 outputs - now fictisious (2 are the same)

# #data_file_name = 'gem_uq_16_std.hdf5'
# #data_file_name = 'gem_uq_79_std.hdf5'
# data_file_name = 'gem_uq_81_std.hdf5'
# features_names_selected = features_names
# target_name_selected = [target_names[1], target_names[1]] # model for [1 - means ; 3- std] of data

# 7) Case from 8 flux tube GEM UQ campaign (4 parameters, tensor product of grid with 2 points per DoF)
#                       and 4 outputs -> in total, output vector of dimensionality 32, as well input dimensionalty of 32

# data_file_name = f"{code_name}_uq_648_transp_std_{index}.hdf5" #gem/gem0 data
# #data_file_name = f"gem0_uq_648_transp_std_{index}.hdf5" #gem0 data!

# features_names_selected = features_names
# target_name_selected = [target_names[0],target_names[1]]

# # 8) Case from 8 flux tube GEM0 5000 runs (4 parameters, tensor product of grid with 5 points per DoF)

n_samples = 5000
#data_file_name = f"{code_name}_{n_samples}_transp_{index}.hdf5" #gem/gem0 data
data_file_name = f"{code_name}_{n_samples}_transp_{index}_{date_gen}.hdf5" #gem/gem0 data

# # 9) Case from 8 flux tube GEM0 8000 runs (4 parameters, 8 flux tubes, 10**3 LHC samples per flux tube)

# n_samples = 8000

# data_file_name = f"{code_name}_{n_samples}_transp_{index}_{date_gen}.hdf5"

###
features_names_selected = features_names
target_name_selected = [target_names[0],target_names[1]]

# === Create a surrogate and its model; train and save it

# Create a campaign object
data_frame = campaign.load_hdf5_data(file_path=data_file_name)
#print(f">train_model, data_frame={data_frame}") ###DEBUG

# Prepare lists of features and array of targets
features = [data_frame[k] for k in features_names_selected if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_name_selected if k in data_frame], axis=1)

time_init_start = t.time()

gp_param = {
            'backend': 'scikit-learn', #'local'
            'process_type': 'gaussian', #'student_t'
            'kernel': 'RBF', #'Matern'
            'length_scale': 1.0,  #[1.]*len(features),
            'noize': 0.1,
            'nu_matern': 1.5,
            'nu_stp': 10,
            'bias': 0.,
            'nonstationary': False,
            'test_frac': 0., #0.2,
            'n_iter': 5,
           }

surrogate = es.methods.GP_Surrogate(
                            backend=gp_param['backend'],
                            n_in=len(features),
                            n_out=target.shape[1],
                                   )

print('Time to initialise the surrogate: {:.3} s'.format(t.time() - time_init_start))

#print(f"> target from train_model: \n {target}") ###DEBUG

time_train_start = t.time()

#print(f">train_model, target={target}") ###DEBUG
surrogate.train(features, 
                target, 
                test_frac=gp_param['test_frac'],
                n_iter=gp_param['n_iter'],
                bias=gp_param['bias'],
                length_scale=gp_param['length_scale'],
                noize=gp_param['noize'],
                nu_matern=gp_param['nu_matern'],
                nu_stp=gp_param['nu_stp'],
                nonstationary=gp_param['nonstationary'],
                process_type=gp_param['process_type'],
                kernel=gp_param['kernel'],
               )

print('Time to train the surrogate: {:.3} s'.format(t.time() - time_train_start))
surrogate.model.print_model_info()

date_str = datetime.now().strftime("%Y%m%d")
save_model_file_name = f"model_{code_name}_val_{gp_param['backend']}{gp_param['process_type']}{gp_param['kernel']}_transp_{index}_{date_str}.pickle"

campaign.add_app(name='gp_campaign', surrogate=surrogate)
campaign.save_state(file_path=save_model_file_name)

# Sequential Design Modification
if SEQDES:
    surrogate.train_sequentially(features, target, n_iter=10, save_history=True)
    surrogate.model.print_model_info()
    campaign.add_app(name='gp_campaign_sequential', surrogate=surrogate)
    campaign.save_state()
