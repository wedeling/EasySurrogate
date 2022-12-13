"""
The script re-initializes an instance of a GPR surrogate and runs an AL algorithm 
(given target QoI value y^*, return an original simulator input vector x^* for which the simulator ouput should be equal y^*)
and returns a set of suggested simulator inputs, 
then re-initializes a EasyVVUQ campaign and sets sampler to run simulator for new cases,
perfroms the runs and retrieves the new simualtor output 
"""

import os
import numpy as np
import chaospy as cp

import easysurrogate as es
import easyvvuq as uq

from easyvvuq.actions import CreateRunDirectory, Encode, Decode, ExecuteLocal, Actions

# the absolute path of this file
HOME = os.path.abspath(os.path.dirname(__file__))

###############################
### EasyVVUQ Campaign - Old ###
###############################

# number of uncertain parameters
D = 5

# Define parameter space
params = {}
for i in range(D):
    params["x%d" % (i + 1)] = {"type": "float",
                               "min": 0.0,
                               "max": 1.0,
                               "default": 0.5}
params["D"] = {"type": "integer", "default": D}
params["out_file"] = {"type": "string", "default": "output.csv"}
output_filename = params["out_file"]["default"]
output_columns = ["f"]

# create encoder, decoder, and execute locally
encoder = uq.encoders.GenericEncoder(template_fname=HOME + '/model/g_func.template',
                                     delimiter='$',
                                     target_filename='in.json')
decoder = uq.decoders.SimpleCSV(target_filename=output_filename,
                                output_columns=output_columns)
execute = ExecuteLocal('{}/model/g_func.py in.json'.format(os.getcwd()))
actions = Actions(CreateRunDirectory('/tmp'),
                  Encode(encoder), execute, Decode(decoder))

# uncertain variables
vary = {}
for i in range(D):
    vary["x%d" % (i + 1)] = cp.Uniform(0, 1)

# MC sampler
my_sampler = uq.sampling.MCSampler(vary=vary, n_mc_samples=50)

# EasyVVUQ Campaign
uq_campaign = uq.Campaign(name='g_func', params=params, actions=actions)

# Associate the sampler with the campaign
uq_campaign.set_sampler(my_sampler)

# Execute runs
uq_campaign.execute().collate()

# get the EasyVVUQ data frame
data_frame = uq_campaign.get_collation_result()

##################################
### EasySurrogate GPR Campaign ###
##################################

# Number of training restarts and fraction of the data to be kept apart for testing
n_iter = 10
test_frac = 0.5

# Create an EasySurrogate campaign
es_campaign = es.Campaign()

# This is the main point of this test: extract training data from EasyVVUQ data frame
features, samples = es_campaign.load_easyvvuq_data(uq_campaign, qoi_cols='f')

# Create gaussian process regression surrogate
surrogate = es.methods.GP_Surrogate(n_in=D)

# Train the GPR
surrogate.train(features, 
                samples['f'], 
                n_iter=n_iter,
                test_frac=test_frac,
                )

# get some useful dimensions of the GPR surrogate
dims = surrogate.get_dimensions()

# evaluate the GPR surrogate on the test data
test_predictions = np.zeros([dims['n_test'], dims['n_out']])
for count, i in enumerate(range(dims['n_train'], dims['n_samples'])):
    test_predictions[count] = surrogate.predict(features[i].reshape(-1,1))[0]

# produce a sample close to average predicted value
feature_names = ["x%d" % (i + 1) for i in range(D)]
qoi_targ = test_predictions.mean()

print('> Looking for samples to yield: {0}'.format(qoi_targ))

x_new = surrogate.train_sequentially(n_iter=1, feats=feature_names, target=qoi_targ, 
                                                acquisition_function='poi_sq_dist_to_val') 

qoi_pred = surrogate.predict(x_new.reshape(-1, 1))[0][0]

print('> The new sample in input space is : {0} and the predicted value is : {1}'.format(
    x_new, qoi_pred))


###############################
### EasyVVUQ Campaign - New ###
###############################

# check the existing EasyVVUQ campaign

camp_name = uq_campaign.campaign_name
#uq_db_file_name = 'uq_campaign.db'
#uq_campaign = uq.Campaign(db_location=uq_db_file_name)
#TODO make a verison of previous initializing from scratch

runs_list_old = uq_campaign.list_runs()
print('> Number of existing runs is: {0}'.format(len(runs_list_old)))
print('> Last of existing runs is : \n {0}'.format(runs_list_old[-1]))

# add new run to the camapign database

# next line is what does the job - only works for a simple case
es_campaign.add_samples_to_easyvvuq(x_new, uq_campaign, feature_names)

runs_list_new = uq_campaign.list_runs()
n_sample_new = len(runs_list_new)
print('> Number of runs including the new one is: {0}'.format(n_sample_new))
print('> Last run including the new one is : \n {0}'.format(runs_list_new[-1]))

# possibly update other parts of the campaign

#sampler = uq_campaign.get_active_sampler()
#uq_campaign.set_sampler(sampler, update=True)
##TODO: may be sampler need to change - data base does not know about samples - it actually should be updated in the .execute()

#actions = Actions()
#uq_campaign.replace_actions(app_name=camp_name, actions=actions)
## TODO: may be actions has to renewed

#app = uq_campaign.get_active_app()
#uq_campaign.set_app(camp_name)

# execute runs for the new sample and colalte data

uq_campaign.execute().collate() 
#TODO test how many samples were executed in this call, must be 1 - there is a STATUS check in .apply_for_each_sample()
data_frame_new = uq_campaign.get_collation_result()

# check what are the results for the new run

f_new = data_frame_new.iloc[n_sample_new-1]['f'].values[0]
print('> Actual value for a new sample is: {0} compared to anticipated: {1}'.format(f_new, qoi_targ))
