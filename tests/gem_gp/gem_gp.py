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
campaign = es.Campaign(load_state=True, file_path='gem_gp_model_tifl.pickle')
data_frame = campaign.load_hdf5_data(file_path='gem_data_625.hdf5')

features = [data_frame[k] for k in features_names_selected if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_name_selected if k in data_frame], axis=1)

feat_train, targ_train, feat_test, targ_test = campaign.surrogate.feat_eng.get_training_data(features, target, index=campaign.surrogate.feat_eng.train_indices)
analysis = es.analysis.GP_analysis(campaign.surrogate)
analysis.get_regression_error(feat_test, targ_test)

r = analysis.auto_correlation_function(targ_test, 5)
print('Te flux auto-correlation is {}'.format(r))

c = analysis.cross_correlation_function(targ_test, analysis.y_pred, 5)
print('Te flux cross-correlation between simulated and predicted value is {}'.format(c))

tefl_dom_tr, tefl_pdf_tr = analysis.get_pdf(targ_test)
tefl_dom_ts, tefl_pdf_ts = analysis.get_pdf(targ_test)
tefl_dom_tot, tefl_pdf_tot = analysis.get_pdf(data_frame[target_name_selected[0]])

tefl_dom_surr, tefl_pdf_surr = analysis.get_pdf(analysis.y_pred)
analysis.plot_pfds(tefl_dom_ts, tefl_pdf_ts, tefl_dom_surr, tefl_pdf_surr, tefl_dom_tr, tefl_pdf_tr, tefl_dom_tot, tefl_pdf_tot)
