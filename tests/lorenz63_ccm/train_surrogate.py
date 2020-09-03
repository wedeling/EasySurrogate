import numpy as np
import easysurrogate as es

#create EasySurrogate campaign
campaign = es.Campaign()

#load HDF5 data frame
data_frame = campaign.load_hdf5_data()

#supervised training data set
feats = [data_frame['X']]
target = data_frame['Y']

#create a CCM surrogate
surrogate = es.methods.CCM_Surrogate()

#Number of time lags
lags = [[1, 10]]
#number of 1D bins per time-lagged feature
N_bins = [10, 10]

#train the CCM
surrogate.train(feats, target, N_bins, lags, test_frac=0.1)

#add the app + save ccm campaign
campaign.add_app(name='CCM_test', surrogate=surrogate)
campaign.save_state()

analysis = es.analysis.CCM_analysis(surrogate)
analysis.plot_2D_binning_object()
analysis.plot_2D_shadow_manifold()
analysis.compare_convex_hull_volumes()