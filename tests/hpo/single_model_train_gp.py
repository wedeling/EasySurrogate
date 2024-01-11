#from py import process
import easysurrogate as es
import sys
import json
import numpy as np

print('> Entering the training script')
rs = 42 # TODO should be passed
np.random.seed(rs)

# Read the current hyperparameter values
json_input = sys.argv[1]
with open(json_input, "r") as f:
    inputs = json.load(f)

# Read the flux tube number
ft = sys.argv[2]

print('Inputs: {0}'.format(inputs))

# Input and output names
features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux', 'te_transp_flux_std', 'ti_transp_flux_std']
features_names_selected = [features_names[0], features_names[1], features_names[2], features_names[3]]

# Chose target value - train for 2 QoIs
#target_name_selected = [target_names[1]]
target_name_selected = [target_names[0], target_names[1]]

# Create EasySurrogate campaign
campaign = es.Campaign()

# Load HDF5 data frame
data_file_name = f"gem06_f{ft}.hdf5"
data_frame = campaign.load_hdf5_data(file_path='../../../'+data_file_name)
# TODO: get rid of hardcoding relative path

# Supervised training data set
features = [data_frame[k] for k in features_names_selected if k in data_frame]
target = np.concatenate([data_frame[k] for k in target_name_selected if k in data_frame], axis=1)

# Create a GPR surrogate
surrogate = es.methods.GP_Surrogate(
    backend=str(inputs['backend']),
    n_in=len(features_names_selected),
    n_out=len(target_name_selected),
    )

#TODO mind if need to restore default value for a parameter
surrogate.train(
    features, 
    target,  
    n_iter=int(inputs['n_iter']),
    test_frac=float(inputs['testset_fraction']),
    kernel=inputs['kernel'],
    length_scale=float(inputs['length_scale']),
    noize=float(inputs['noize']),
    bias=float(inputs['bias']),
    nu_matern=float(inputs['nu_matern']),
    nu_stp=float(inputs['nu_stp']),
    process_type=str(inputs['process_type'])
    )

surrogate.model.print_model_info()

campaign.add_app(name='gp_campaign', surrogate=surrogate)
campaign.save_state(file_path='model.pickle')

# Performing surrogate analysis, here: measuring perfromance on testing/validation data
feat_train, targ_train, feat_test, targ_test = campaign.surrogate.feat_eng.\
    get_training_data(features, target, index=campaign.surrogate.feat_eng.train_indices)

analysis = es.analysis.GP_analysis(
                        gp_surrogate=campaign.surrogate,
                        target_name_selected=target_name_selected,
                        features_names_selected=features_names_selected,
                        nft=ft,
                                  )

err_test_abs, err_test, r2_test = analysis.get_regression_error(
    feat_test,
    targ_test,
    feat_train,
    targ_train,
    flag_plot=True,
    addit_name=str(ft),
    remainder_file_path='../../../scan_gem0_remainder_',
    remainder_file_date='20240110',
    )

# TODO return analysis for 2 QoIs and procude a compound loss

err_test_tot = [float(np.abs(err).mean()) for err in err_test]

loss_list = [1. - r2 for r2 in r2_test]

loss = sum(loss_list) / len(loss_list)

# Writing an output
output = {'loss': loss}
with open('output.json', 'w') as of:
    json_string = json.dumps(output)
    of.write(json_string)

print(f"Outputs: {output}")

print(f"> Exiting the training script")
