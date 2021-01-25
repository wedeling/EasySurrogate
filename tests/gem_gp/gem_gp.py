import os
import sys
import timeit
import matplotlib.pyplot as plt
import numpy as np
import csv

import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux']

### PREPARING MODEL TO USE
#load pre-trained campaign
campaign = es.Campaign(load_state=True, file_path='gem_gp_model.pickle')
data_frame = campaign.load_hdf5_data(file_path='gem_data_625.hdf5')
#features = [np.array(data_frame[k]).transpose().reshape((-1, 1)) for k in features_names if k in data_frame]
#target = np.array([data_frame[k] for k in target_names if k in data_frame]).transpose()

analysis = es.analysis.GP_analysis(campaign.surrogate)
analysis.get_regression_error()
# TODO add more in analysis
