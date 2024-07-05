import time as t
import numpy as np
import sys
from datetime import datetime

import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho', 'profiles_1d_q', 'profiles_1d_gm3']
target_names = ['te_transp_flux', 'ti_transp_flux', 'te_transp_flux_std', 'ti_transp_flux_std']

np.random.seed(42)

SEQDES = False #True if Sequential Design of Experiments to be used

if len(sys.argv) < 2 :
    index = 0
else:
    index = sys.argv[1]

if len(sys.argv) < 3 :
    date_gen = datetime.now().strftime("%Y%m%d")
else:  
    date_gen = sys.argv[2]

if len(sys.argv) < 4 :
    model_id = datetime.now().strftime("%Y%m%d")
else:
    model_id = sys.argv[3]

code_name = sys.argv[4] if len(sys.argv)>4 else 'gem0py'

campaign = es.Campaign(load_state=False)

# # I) Case from 8 flux tube GEM0 5000 runs (4 parameters, tensor product of grid with 5 points per DoF)

# n_samples = sys.argv[5] if len(sys.argv)>5 else 5000

# n_params = 4

# II) Case w/ 8 f-t-s pyGEM0 runs, possibly equilibrium included, full tensor product

# III) Same as (II) but all quantities are in log (base model is Q = Q_0 * x^G )
features_names = ['te_value_ln', 'ti_value_ln', 'te_ddrho_ln', 'ti_ddrho_ln',]
target_names = ['te_transp_flux_ln', 'ti_transp_flux_ln']

# TODO should be deducible from the read data! e.g. from the .hdf5 file
n_samples = int(sys.argv[5]) if len(sys.argv)>5 else 5832

n_params  = int(sys.argv[6]) if len(sys.argv)>6 else 6

###
data_file_name = f"{code_name}_{n_samples}_transp_{index}_{date_gen}.hdf5"

features_names_selected = features_names[0:n_params]
target_name_selected = [target_names[0],target_names[1]]

# === Create a surrogate and its core model behind the object; train and save it

# Create a campaign object
data_frame = campaign.load_hdf5_data(file_path=data_file_name)
#print(f">train_model, data_frame={data_frame}") ###DEBUG
#print(f"{data_frame.keys()}") ###DEBUG

# Prepare lists of features and array of targets
features = [data_frame[k] for k in features_names_selected if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_name_selected if k in data_frame], axis=1)

#print(f"len features:\n{len(features)}") ###DEBUG
time_init_start = t.time()

gp_param = {
            'backend': 'scikit-learn', #'local'
            'process_type': 'gaussian', #'student_t'
            'kernel': 'RBF', #'Matern'
            'length_scale': 1.0,  #[1.]*len(features),
            'length_scale_bounds': (1e-1, 1e+6),
            'noize': 0.1,
            'noize_bounds': (1e-3, 1e+3),
            'nu_matern': 1.5,
            'nu_stp': 10,
            'bias': 0.,
            'nonstationary': False,
            'test_frac': 0., #0.2,
            'n_iter': 20,
           }

surrogate = es.methods.GP_Surrogate(
                            backend=gp_param['backend'],
                            n_in=len(features),
                            n_out=target.shape[1],
                                   )

print('Time to initialise the surrogate: {:.3} s'.format(t.time() - time_init_start))
#print(f"> target from train_model: \n {target}") ###DEBUG

time_train_start = t.time()

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
                length_scale_bounds=gp_param['length_scale_bounds'],
                noize_bounds=gp_param['noize_bounds'],
               )

print('Time to train the surrogate: {:.3} s'.format(t.time() - time_train_start))
surrogate.model.print_model_info()

save_model_file_name = f"model_{code_name}_val_{gp_param['backend']}{gp_param['process_type']}{gp_param['kernel']}_transp_{index}_{model_id}.pickle"

campaign.add_app(name='gp_campaign', surrogate=surrogate)
campaign.save_state(file_path=save_model_file_name)

# Sequential Design Modification
if SEQDES:
    surrogate.train_sequentially(features, target, n_iter=10, save_history=True)
    surrogate.model.print_model_info()
    campaign.add_app(name='gp_campaign_sequential', surrogate=surrogate)
    campaign.save_state()
