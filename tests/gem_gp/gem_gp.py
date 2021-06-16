import time as t
import numpy as np

import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux']

features_names_selected = features_names
target_name_selected = target_names

### PREPARING MODEL TO USE
#load pre-trained campaign

# 1) case for data from single flux tube GEM UQ campaign
#campaign = es.Campaign(load_state=True, file_path='gp_gem_625.pickle')
#data_frame = campaign.load_hdf5_data(file_path='gem_data_625.hdf5')
#data_frame_train = campaign.load_hdf5_data(file_path='gem_data_625.hdf5')

# 2) case for data from a MFW production run
#campaign = es.Campaign(load_state=True, file_path='skl_gem_500_wf_1405_opt.pickle')
#data_frame = campaign.load_hdf5_data(file_path='gem_workflow_500.hdf5')
#data_frame_train = campaign.load_hdf5_data(file_path='gem_workflow_500.hdf5')

# 3) case for data generated from single flux tube GEM0 with 4 parameters (LHD, with a wrapper)
#campaign = es.Campaign(load_state=True, file_path='gem0_lhc.pickle')
#data_frame = campaign.load_hdf5_data(file_path='gem_workflow_500.hdf5')
#data_frame_train = campaign.load_hdf5_data(file_path='gem_workflow_500.hdf5')

# 4) case for from single flux tube GEM0 with 2 parameters (LHD, with a wrapper)
features_names_selected = [features_names[2], features_names[3]]
target_name_selected = [target_names[1]]
campaign = es.Campaign(load_state=True, file_path='gp_model_10pperctrset_plus1oseqsamples.pickle')
data_frame = campaign.load_hdf5_data(file_path='gem0_lhc_256.hdf5')
data_frame_train = campaign.load_hdf5_data(file_path='gem0_lhc_256.hdf5')

# getting the data
features_train = [data_frame_train[k] for k in features_names_selected if k in data_frame_train]
target_train = np.concatenate([data_frame_train[k] for k in target_name_selected if k in data_frame_train], axis=1)

feat_train, targ_train, feat_test, targ_test = campaign.surrogate.feat_eng.\
    get_training_data(features_train, target_train, index=campaign.surrogate.feat_eng.train_indices)

features = [data_frame[k] for k in features_names_selected if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_name_selected if k in data_frame], axis=1)

# create analysis class
analysis = es.analysis.GP_analysis(campaign.surrogate)

### SEQ DES
analysis.plot_2d_design_history(x_test=feat_test, y_test=targ_test)

#analysis.get_regression_error(np.concatenate([feat_train, feat_test], axis=0), np.concatenate([targ_train, targ_test], axis=0))
analysis.get_regression_error(feat_test, targ_test, feat_train, targ_train, )

# Cross-correlation functions
r = analysis.auto_correlation_function(targ_test, 5)
print('Te flux auto-correlation is {}'.format(r))

c = analysis.cross_correlation_function(targ_test, analysis.y_pred, 5)
print('Te flux cross-correlation between simulated and predicted value is {}'.format(c))

# Distribution of output variables

tefl_dom_tr, tefl_pdf_tr = analysis.get_pdf(targ_train)
tefl_dom_ts, tefl_pdf_ts = analysis.get_pdf(targ_test)
tefl_dom_tot, tefl_pdf_tot = analysis.get_pdf(data_frame[target_name_selected[0]])
tefl_dom_surr, tefl_pdf_surr = analysis.get_pdf(analysis.y_pred)

analysis.plot_pdfs(tefl_dom_ts, tefl_pdf_ts, tefl_dom_surr, tefl_pdf_surr)

# Distributions of input variables
tevl_dom_tr, tevl_pdf_tr = analysis.get_pdf(feat_train[0])
tevl_dom_ts, tevl_pdf_ts = analysis.get_pdf(feat_test[0])
tivl_dom_tr, tivl_pdf_tr = analysis.get_pdf(feat_train[1])
tivl_dom_ts, tivl_pdf_ts = analysis.get_pdf(feat_test[1])

analysis.plot_pdfs(tevl_dom_tr, tevl_pdf_tr, tevl_dom_ts, tevl_pdf_ts, tivl_dom_tr, tivl_pdf_tr, tivl_dom_ts, tivl_pdf_ts,
                   names=['tevl tr', 'tevl ts', 'tivl tr', 'tivl ts'])