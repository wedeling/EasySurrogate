import numpy as np
import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['ti_transp_flux']

qoi_targ=2099023.289881937 #TODO read from production run result csv

campaign = es.Campaign(load_state=True, file_path='model_val_15112022.pickle')
data_frame = campaign.load_hdf5_data(file_path='gem_uq_81_std.hdf5')

m, v = campaign.surrogate.predict(campaign.surrogate.X_test[1, :].reshape(-1,1))  # just a test on single sample
print(m, v)

features = [data_frame[k] for k in features_names if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_names if k in data_frame], axis=1)

# try to retrain on probability of improvement to be closer to a target QoI value using samples of full feature dataset
X_new = campaign.surrogate.train_sequentially(n_iter=1, feats=features_names, target=qoi_targ)
campaign.save_state(file_path='model_val_seq_16112022.pickle')

#X_new = campaign.surrogate.x_scaler.inverse_transform(X_new)

m, v = campaign.surrogate.predict(X_new.reshape(-1, 1))  # just a test on single sample
print(m, v)