import time as t
import numpy as np
import sys

import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho', 'profiles_1d_q', 'profiles_1d_gm3']
target_names = ['te_transp_flux', 'ti_transp_flux', 'te_transp_flux_std', 'ti_transp_flux_std']

if len(sys.argv) < 2 :
    index = 0
else:
    index = sys.argv[1]

if len(sys.argv) < 3 :
    model_date = '20231016'
else:
    model_date = sys.argv[2]

if len(sys.argv) < 4 :
    data_date = '20231216'
else:
    data_date = sys.argv[3]

if len(sys.argv) < 5 :
    scan_date = '20240110'
else:
    scan_date = sys.argv[4]

code_name = sys.argv[5] if len(sys.argv)>5 else'gem0py'

features_names_selected = features_names
target_name_selected = target_names

# PREPARING MODEL TO USE
# load pre-trained campaign

# # I) Case from 8 flux tubes GEM0 UQ campaign (4 parameters, tensor product of grid with 5 points per DoF)
# n_samples = 5000
# n_params = 4

# II) Case w/ 8 f-t-s pyGEM0 runs, possibly equilibrium included, full tensor product

n_samples = int(sys.argv[6]) if len(sys.argv)>6 else 5832

n_params  = int(sys.argv[7]) if len(sys.argv)>7 else 6

###
backend='scikit-learn'
likelihood='gaussian'
kernel='RBF'

saved_model_file_path = f"model_{code_name}_val_{backend}{likelihood}{kernel}_transp_{index}_{model_date}.pickle"

data_file = f"{code_name}_{n_samples}_transp_{index}_{data_date}.hdf5"

###
features_names_selected = features_names[:n_params]
target_name_selected = [target_names[0], target_names[1]]

# Creating campaign

campaign = es.Campaign(load_state=True, file_path=saved_model_file_path)

data_frame = campaign.load_hdf5_data(file_path=data_file)
data_frame_train = campaign.load_hdf5_data(file_path=data_file)

# To use or not to use analysis of sequential design results
SEQDES = False

# Getting the data
features_train = [data_frame_train[k] for k in features_names_selected if k in data_frame_train]
target_train = np.concatenate([data_frame_train[k]
                              for k in target_name_selected if k in data_frame_train], axis=1)

feat_train, targ_train, feat_test, targ_test = campaign.surrogate.feat_eng.\
    get_training_data(features_train, target_train, index=campaign.surrogate.feat_eng.train_indices)

features = [data_frame[k] for k in features_names_selected if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_name_selected if k in data_frame], axis=1)

# Create analysis class
analysis = es.analysis.GP_analysis(campaign.surrogate,
                                   target_name_selected=target_name_selected,
                                   features_names_selected=features_names_selected,
                                   nft=index,
                                  )

# SEQ DES
if SEQDES:
    analysis.plot_2d_design_history(x_test=feat_test, y_test=targ_test)

analysis.get_regression_error(feat_test, targ_test, feat_train, targ_train, 
                              #index=[i for i in range(16)] #DEBUG
                              addit_name=str(index),
                              remainder_file_date=scan_date,
                             )

# Cross-correlation functions
"""
print('targ_test={0}'.format(targ_test)) ###DEBUG
r = analysis.auto_correlation_function(targ_test, len(targ_test) if len(targ_test) < 5 else 5)
print('Ti transport flux auto-correlation is {}'.format(r))

c = analysis.cross_correlation_function(targ_test, analysis.y_pred, len(targ_test) if len(targ_test) < 5 else 5)
print('Ti transport flux cross-correlation between simulated and predicted value is {}'.format(c))
"""

# Distribution of output variables
"""
tefl_dom_tr, tefl_pdf_tr = analysis.get_pdf(targ_train)
tefl_dom_ts, tefl_pdf_ts = analysis.get_pdf(targ_test)
tefl_dom_tot, tefl_pdf_tot = analysis.get_pdf(data_frame[target_name_selected[0]])
tefl_dom_surr, tefl_pdf_surr = analysis.get_pdf(analysis.y_pred)

analysis.plot_pdfs(tefl_dom_ts, tefl_pdf_ts, tefl_dom_surr, tefl_pdf_surr)
"""

# Distributions of input variables
"""
tevl_dom_tr, tevl_pdf_tr = analysis.get_pdf(feat_train[0])
tevl_dom_ts, tevl_pdf_ts = analysis.get_pdf(feat_test[0])
tivl_dom_tr, tivl_pdf_tr = analysis.get_pdf(feat_train[1])
tivl_dom_ts, tivl_pdf_ts = analysis.get_pdf(feat_test[1])

analysis.plot_pdfs(
    tevl_dom_tr,
    tevl_pdf_tr,
    tevl_dom_ts,
    tevl_pdf_ts,
    tivl_dom_tr,
    tivl_pdf_tr,
    tivl_dom_ts,
    tivl_pdf_ts,
    names=[
        'tevl tr',
        'tevl ts',
        'tivl tr',
        'tivl ts'])
"""
