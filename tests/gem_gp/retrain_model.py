import numpy as np
import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux']

campaign = es.Campaign(load_state=True, file_path='gem_gp_model.pickle')
data_frame = campaign.load_hdf5_data(file_path='gem_data_625.hdf5')

m, v = campaign.surrogate.predict(campaign.surrogate.X_test[100,:])
print(m, v)

features = [data_frame[k] for k in features_names if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_names if k in data_frame], axis=1)

campaign.surrogate.train(features, target, test_frac=0.0, postrain=True)  # try to retrain on max uncertain samples of full feature dataset
#campaign.save_state()
m, v = campaign.surrogate.predict(campaign.surrogate.X_test[100,:])
print(m, v)
