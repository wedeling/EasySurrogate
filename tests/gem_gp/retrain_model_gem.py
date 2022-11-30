import numpy as np
import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['ti_transp_flux']

qoi_targ=2099023.289881937 #TODO read from production run result csv

campaign = es.Campaign(load_state=True, file_path='model_val_LocGaussMatern_30112022.pickle')
data_frame = campaign.load_hdf5_data(file_path='gem_uq_81_std.hdf5')

# Printing out some predictions from the existing model

# for X_i in campaign.surrogate.X_test:
#      print(campaign.surrogate.x_scaler.inverse_transform(X_i.reshape(1,-1))) 

# test_res = [campaign.surrogate.predict(campaign.surrogate.x_scaler.inverse_transform(X_i.reshape(1,-1)).reshape(-1,1)) for X_i in campaign.surrogate.X_test]
# print(test_res)

# Preparing new features and targets
features = [data_frame[k] for k in features_names if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_names if k in data_frame], axis=1)

# Try to retrain on probability of improvement to be closer to a target QoI value using samples of full feature dataset
X_new = campaign.surrogate.train_sequentially(n_iter=1, feats=features_names, target=qoi_targ, acquisition_function='poi_sq_dist_to_val') #'poi_sq_dist_to_val'
#campaign.save_state(file_path='model_val_new_16112022.pickle')

#X_new = campaign.surrogate.x_scaler.inverse_transform(X_new)

# m_new, v_new = campaign.surrogate.predict(X_new.reshape(-1, 1))  # just a test on single sample
# print(m_new, v_new)