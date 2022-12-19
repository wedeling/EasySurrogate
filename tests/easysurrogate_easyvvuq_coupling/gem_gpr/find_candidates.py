import numpy as np
import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['ti_transp_flux']

qoi_targ=2099023.289881937 #TODO read from production run result csv
print('> Looking for samples to yield: {0}'.format(qoi_targ))

campaign = es.Campaign(load_state=True, file_path='model_val_LocStudentMatern_19122022.pickle')
data_frame = campaign.load_hdf5_data(file_path='gem_uq_81_std.hdf5')

# Preparing new features and targets
features = [data_frame[k] for k in features_names if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_names if k in data_frame], axis=1)

# Try to retrain on probability of improvement to be closer to a target QoI value using samples of full feature dataset
X_new = campaign.surrogate.train_sequentially(
                            n_iter=1, # crucial, otherwise will try to start re-training model
                            feats=features_names, 
                            target=qoi_targ, 
                            acquisition_function='poi_sq_dist_to_val',
                                             )

# Test on a single sample
qoi_pred = campaign.surrogate.predict(X_new.reshape(-1, 1))[0][0]
m_new, v_new = campaign.surrogate.predict(X_new.reshape(-1, 1))

print('> The new sample in input space is : {0} and the predicted value is : {1}'.format(m_new, qoi_pred))
