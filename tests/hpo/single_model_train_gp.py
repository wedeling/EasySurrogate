import easysurrogate as es
import sys
import json
import numpy as np

print('> Entering the training script')

# read the current hyperparameter values
json_input = sys.argv[1]
with open(json_input, "r") as f:
    inputs = json.load(f)

# input and output names
features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux']
features_names_selected = [features_names[0], features_names[1], features_names[2], features_names[3]]
target_name_selected = [target_names[1]]

# create EasySurrogate campaign
campaign = es.Campaign()

# load HDF5 data frame
data_frame = campaign.load_hdf5_data(file_path='../../../gem.hdf5')
# TODO: get rid of hardcoding relative path

# supervised training data set
features = [data_frame[k] for k in features_names_selected if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_name_selected if k in data_frame], axis=1)

# create a GP surrogate
surrogate = es.methods.GP_Surrogate(
    n_in=4,
    )

surrogate.train(
    features, 
    target,  
    n_iter=10,
    test_frac=0.5,
    kernel='Matern',
    length_scale=float(inputs['length_scale']),
    noize=float(inputs['noize']),
    bias=float(inputs['bias'])
    )

campaign.add_app(name='gp_campaign', surrogate=surrogate)
campaign.save_state(file_path='model.pickle')

# performing surrogate analysis, here: measuring perfromance on testing/validation data
feat_train, targ_train, feat_test, targ_test = campaign.surrogate.feat_eng.\
    get_training_data(features, target, index=campaign.surrogate.feat_eng.train_indices)

analysis = es.analysis.GP_analysis(gp_surrogate=surrogate)
err_test_abs, err_test,= analysis.get_regression_error(
    feat_test,
    targ_test,
    feat_train,
    targ_train,
    flag_plot=False,
    )
#TODO: check how training set is generated, and how error is defined

err_test_tot = float(np.abs(err_test).mean())

# writing an output
output = {'test_error': err_test_tot}
with open('output.json', 'w') as of:
    json_string = json.dumps(output)
    of.write(json_string)

print('> Exiting the training script')
