import time as t
import numpy as np

import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux', 'te_transp_flux_std', 'ti_transp_flux_std']

features_names_selected = features_names
target_name_selected = target_names

# PREPARING MODEL TO USE
# load pre-trained campaign

# 6) Cases predicted by AL GPR model, the values should yield results close to 2099023.289881937 
saved_model_file_path = 'model_val_LocStudentMatern_13012023.pickle'

features_names_selected = features_names
target_name_selected = [target_names[1]]
campaign = es.Campaign(load_state=True, file_path=saved_model_file_path)

#data_frame = campaign.load_hdf5_data(file_path='gpr_al_6_val.hdf5')
data_frame = campaign.load_hdf5_data(file_path='gem_uq_81_std.hdf5')
data_frame_train = campaign.load_hdf5_data(file_path='gem_uq_81_std.hdf5')

# Getting the data
features_train = [data_frame_train[k] for k in features_names_selected if k in data_frame_train]
target_train = np.concatenate([data_frame_train[k]
                              for k in target_name_selected if k in data_frame_train], axis=1)

feat_train, targ_train, feat_test, targ_test = campaign.surrogate.feat_eng.\
    get_training_data(features_train, target_train, index=campaign.surrogate.feat_eng.train_indices)

features = [data_frame[k] for k in features_names_selected if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_name_selected if k in data_frame], axis=1)

# Create analysis class
analysis = es.analysis.GP_analysis(campaign.surrogate)

#analysis.get_regression_error(np.concatenate([feat_train, feat_test], axis=0), np.concatenate([targ_train, targ_test], axis=0))

analysis.get_regression_error(feat_test, targ_test, feat_train, targ_train,)
