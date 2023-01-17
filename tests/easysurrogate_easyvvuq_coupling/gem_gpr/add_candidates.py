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

import csv

import easysurrogate as es
import easyvvuq as uq

from easyvvuq.actions import CreateRunDirectory, Encode, Decode, ExecuteLocal, Actions

# The absolute path of this file
HOME = os.path.abspath(os.path.dirname(__file__))

###############################
### EasyVVUQ Campaign - New ###
###############################

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['ti_transp_flux']

es_campaign = es.Campaign(load_state=True, file_path='model_val_LocGaussianMatern_13012023.pickle')

qoi_targ=2099023.289881937 #TODO: read from prevous script or campaign files

# Read the samples to be added
#TODO: following block should probably be a function in EasyVVUQ.Camapign()

params_names = [s.replace('_', '.') for s in features_names]
param_lookup_dict = {s:params_names[i] for i,s in enumerate(features_names)}

samples_file = 'surrogate_al_cands.csv'

with open(samples_file, 'r') as f: 
    runs_new = [
        {param_lookup_dict[k]:float(v) for k,v in row.items() if k!=''} 
            for row in csv.DictReader(f, skipinitialspace=True)
               ]

print('Adding these runs: {0} \n'.format(runs_new))

# Load the existing EasyVVUQ campaign

uq_db_file_name = 'campaign.db'
uq_campaign_name_prefix = 'VARY_1FT_GEM_NT_'
uq_campaign_folder_name = 'akgbbn1a'
uq_db_file_name_full = 'sqlite:///' + uq_campaign_name_prefix + uq_campaign_folder_name + '/' + uq_db_file_name
uq_campaign = uq.Campaign(name=uq_campaign_name_prefix, db_location=uq_db_file_name_full)

runs_list_old = uq_campaign.list_runs()
print('> Number of existing runs is: {0}'.format(len(runs_list_old)))
print('> Last of existing runs is : \n {0}'.format(runs_list_old[-1]))

# Add new runs to the campaign database

# Next line is what does the job - only works for a simple case
uq_campaign.add_runs(runs=runs_new)
#es_campaign.add_samples_to_easyvvuq(x_new, uq_campaign, features_names)

runs_list_new = uq_campaign.list_runs()
n_sample_new = len(runs_list_new)
print('> Number of runs including the new one is: {0}'.format(n_sample_new))
print('> Last run including the new one is : \n {0}'.format(runs_list_new[-1]))

# Check (and possibly update) other parts of the campaign

# Execute runs for the new sample and colalte data

###uq_campaign.execute().collate() #ATTENTION: before here the cluster set up should be done, and all the actions checked

#TODO test how many samples were executed in this call, must be 1 - there is a STATUS check in .apply_for_each_sample()
data_frame_new = uq_campaign.get_collation_result()

# Check what are the results for the new run

f_new = data_frame_new.iloc[n_sample_new-1][target_names[0]].values[0]
print('> Actual value for a new sample is: {0} compared to anticipated: {1}'.format(f_new, qoi_targ))
