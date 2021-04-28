import time as t
import numpy as np

import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux']

#features_names_selected = [features_names[1], features_names[3]]
features_names_selected = features_names
target_name_selected = [target_names[1]]

### PREPARING MODEL TO USE
#load pre-trained campaign
campaign = es.Campaign(load_state=True, file_path='gp_gem_625.pickle')
data_frame = campaign.load_hdf5_data(file_path='gem_workflow_100.hdf5')
#data_frame = campaign.load_hdf5_data(file_path='gem_data_625.hdf5')

features = [data_frame[k] for k in features_names_selected if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_name_selected if k in data_frame], axis=1)


data_frame_train = campaign.load_hdf5_data(file_path='gem_data_625.hdf5')
features_train = [data_frame_train[k] for k in features_names_selected if k in data_frame_train]
target_train = np.concatenate([data_frame_train[k] for k in target_name_selected if k in data_frame_train], axis=1)
feat_train, targ_train, feat_test, targ_test = campaign.surrogate.feat_eng.\
    get_training_data(features_train, target_train, index=campaign.surrogate.feat_eng.train_indices)

#DEBUG
# print(target)
# print(targ_train)
# print(targ_test)
# print(data_frame['ti_transp_flux'].min(), data_frame['ti_transp_flux'].max())
# print(campaign.surrogate.feat_eng.train_indices)
# print(campaign.surrogate.feat_eng.test_indices)
# print(target.shape)
# print(features)
# print(np.concatenate([targ_train, targ_test], axis=0).shape)

features = np.array(features)
print(features.shape)
features = np.moveaxis(features, 0, -1)
features = features.reshape(features.shape[0], -1)
print(features.shape)

analysis = es.analysis.GP_analysis(campaign.surrogate)
#analysis.get_regression_error(feat_test, targ_test)
#analysis.get_regression_error(np.concatenate([feat_train, feat_test], axis=0), np.concatenate([targ_train, targ_test], axis=0))
#analysis.get_regression_error(feat_test, targ_test, feat_train, targ_train)
analysis.get_regression_error(features, target, x_test_inds=np.arange(len(target)))

r = analysis.auto_correlation_function(targ_test, 5)
print('Te flux auto-correlation is {}'.format(r))

#c = analysis.cross_correlation_function(targ_test, analysis.y_pred, 5)
#print('Te flux cross-correlation between simulated and predicted value is {}'.format(c))

tefl_dom_tr, tefl_pdf_tr = analysis.get_pdf(targ_train)
tefl_dom_ts, tefl_pdf_ts = analysis.get_pdf(targ_test)
tefl_dom_tot, tefl_pdf_tot = analysis.get_pdf(data_frame[target_name_selected[0]])

tefl_dom_surr, tefl_pdf_surr = analysis.get_pdf(analysis.y_pred)
analysis.plot_pfds(tefl_dom_ts, tefl_pdf_ts, tefl_dom_surr, tefl_pdf_surr, tefl_dom_tr, tefl_pdf_tr)

# Distributions of input variables
tevl_dom_tr, tevl_pdf_tr = analysis.get_pdf(feat_train[0])
tevl_dom_ts, tevl_pdf_ts = analysis.get_pdf(feat_test[0])
tivl_dom_tr, tivl_pdf_tr = analysis.get_pdf(feat_train[1])
tivl_dom_ts, tivl_pdf_ts = analysis.get_pdf(feat_test[1])

analysis.plot_pfds(tevl_dom_tr, tevl_pdf_tr, tevl_dom_ts, tevl_pdf_ts, tivl_dom_tr, tivl_pdf_tr, tivl_dom_ts, tivl_pdf_ts,
                   names=['tevl tr', 'tevl ts', 'tivl tr', 'tivl ts'])

# OBSERVATION: very sensitive to the initial choice of training dataset, can miss trend
#   if there are no samples close to extrema

# OBSERVATION: forward prediction devireges and yields negative value of fluxes
# the distribution of inputs (Te, Ti) in testing set has support regions that is not covered by traing set