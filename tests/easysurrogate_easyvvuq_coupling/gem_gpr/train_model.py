import time as t
import numpy as np

import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux', 'te_transp_flux_std', 'ti_transp_flux_std']

np.random.seed(42)

campaign = es.Campaign(load_state=False)

# Case from single flux tube GEM UQ campaign (4 parameters, tensor product of grid with 2 points per DoF)
data_file_name = 'gem_uq_81_std.hdf5'
features_names_selected = features_names
target_name_selected = [target_names[1]] # model for [1 - means ; 3- std] of data

# Create a surrogate and its model; train and save it

# Create a campaign object
data_frame = campaign.load_hdf5_data(file_path=data_file_name)

# prepare lists of features and array of targets
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
            'test_frac': 0.5,
            'n_iter': 5,
           }

surrogate = es.methods.GP_Surrogate(
                            backend=gp_param['backend'],
                            n_in=len(features),
                                   )

print('Time to initialise the surrogate: {:.3} s'.format(t.time() - time_init_start))

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
               )

print('Time to train the surrogate: {:.3} s'.format(t.time() - time_train_start))
surrogate.model.print_model_info()

save_model_file_name = 'model_val_SkitGaussianRbf_16012023.pickle'

campaign.add_app(name='gp_campaign', surrogate=surrogate)
campaign.save_state(file_path=save_model_file_name)
