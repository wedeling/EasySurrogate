"""
Run a EasyVVUQ with a surrogate trainging as an application.
Varying parameters are the Hyperparameters of the surrogate models.
"""

import os
import sys
import pickle
import time

import easyvvuq as uq
from easyvvuq.actions import QCGPJPool
from easyvvuq.actions.execute_qcgpj import EasyVVUQParallelTemplate

from qcg.pilotjob.executor_api.qcgpj_executor import QCGPJExecutor

from pprint import pprint
from itertools import product
import csv
import numpy as np
import math as m

import easysurrogate as es

#TODO: ADD A RANDOM SEED
rs = 42
np.random.seed(rs)

# Choice of flux tube
ft = sys.argv[1]

print(f"> Starting a hyperparameter optimisation for flux tube {ft}, random_seed={rs}")

#TODO write down test fraction expliceitly - now it is 0.2 by default

# List all possible hyperparamters of a surrogate of this type, together with their types and default values
params = {
    #"length_scale": {"type": "string", "min": 1e-12, "max": 1e+12, "default": "1.0"},
    #"noize": {"type": "string", "min": 1e-16, "max": 1e+4, "default": "1e-3"}, 
    #"bias": {"type": "string", "min": -1e+6, "max": 1e+6, "default": "0.0"},
    #"backend" : {"type": "string", "default": "local"},
    "testset_fraction": {"type": "string", "min": 0.0, "max": 1.0, "default": "0.2"},
    "n_iter" : {"type": "string", "min": 1, "default": "1000"},
    "n_layers" : {"type": "string", "min": 1, "default": "2"},
    "n_neurons" : {"type": "string", "min": 1, "default": "16"},
    "batch_size" : {"type": "string", "min": 1, "default": "32"},
    "activation" : {"type": "string", "default": "relu"},
}
# looks like EasyVVUQ checks for a need in a default value after sampler is initialized 

# TODO should be read from CSV
# TODO force CSVSampler to interpret entries with correct type (int-s as int-s!)

# For Grid Search: form carthesian product of variables

vary = {} #TODO maybe: use vary to create CSV for non-categorical hyperparameters

# Define values for each parameter, create its cartesian grid, save as csv
param_search_vals = {
    #"length_scale": [0.5, 1.0, 2.0], # not used
    #"noize": [1e-4, 1e-2, 1e-1], # not used
    #"bias": [0., 1.0], # not used
    #"backend" : ['local'] # ['local', 'scikit-learn'], # not used
    "n_iter" : [1000, 2500, 5000, 10000, 20000, 30000, 40000],
    "n_layers" : [2, 3, 4, 5],
    "n_neurons" : [8, 16, 32, 64, 128],
    "batch_size" : [8, 16, 32, 64, 128],
    "activation" : ['relu', 'leaky_relu', 'tanh', 'sigmoid'],
}

csv_header = [k for k in param_search_vals.keys()]
csv_vals = [x for x in product(*[v for (k,v) in param_search_vals.items()])]

def clean_grid_by_rules(header, vals, def_vals):

    # Construct list of dictionaries of with keys taken from header and values taken from rows of vals
    data = [{header[i]:vals[j][i] for i in range(len(header))} for j in range(len(vals))]

    vals_new = []

    for d in data:
        pass
        vals_new.append([x for k,x in d.items()])

    print('> Using {0} different parameter values combinations instead of full {1}'.format(len(vals_new), len(vals)))

    return vals_new

csv_defaults = {k:v['default'] for k,v in params.items()}
#csv_vals_new = clean_grid_by_rules(csv_header, csv_vals, csv_defaults)

with open('hp_values_ann_loc_2.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(csv_vals)

#with open('hp_values_ann_loc_short_1.csv', 'w') as f:
#    writer = csv.writer(f)
#    writer.writerow(csv_header)
#    writer.writerows(csv_vals_new)
 
# If run on HPC, should be called from the scheduler like SLURM
# for which an environmental variable HPC_EXECUTION should be specified
HPC_EXECUTION = os.environ['HPC_EXECUTION']

campaign_name = f"hpo_easysurrogate_ann_f{ft}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_"
work_dir = ''
#TODO specify flexible paths

campaign = uq.Campaign(name=campaign_name, work_dir=work_dir)

# Optimising hyperparamters for GEM data
# Here listing different test cases for different combinations of varied parameters

# A file specifying a table of hyperparameter values (or their ranges/distributions) to pass for a number of ML models
# Mind: delimeter is ',' w/o spaces

param_file = 'hp_values_ann_loc_2.csv'

# Encoder should take a value from the sampler and pass it to EasySurrogate es.methos.*_Surrogate().train(...) as kwargs
encoder = uq.encoders.GenericEncoder(
    template_fname='hpo_ann.template',
    delimiter='$',
    target_filename='input.json'
)

# Decoder should take a training, or better validation, loss values from an ML model for given hyperparameter value
qoi = ['loss']
decoder = uq.decoders.JSONDecoder(
    target_filename='output.json',
    output_columns=qoi,
)

# Execute should train a model in EasySurrogate: get the data, initalise object, call .train() and calculate the training/validation error
execute_train = uq.actions.ExecuteLocal(
    f"python3 ../../../single_model_train_ann.py input.json {ft} > train.log"
)
# TODO get rid of hard-coding relative paths

actions = uq.actions.Actions(
    uq.actions.CreateRunDirectory('/runs', flatten=True),
    uq.actions.Encode(encoder),
    execute_train,
    uq.actions.Decode(decoder),
)

campaign.add_app(
    name=campaign_name,
    params=params, # TODO read from CSV
    actions=actions,
)

# Sampler should read hyperparameter-value dictionary from a CSV file
sampler = uq.sampling.CSVSampler(filename=param_file) 
# TODO: sampler has to read numbers as integers
campaign.set_sampler(sampler)

# Execute: train a number of ML models
print(f"> Starting to train the models")
start_time = time.time()

if HPC_EXECUTION:

    with QCGPJPool(
            #qcgpj_executor=QCGPJExecutor(),
            template=EasyVVUQParallelTemplate(),
            template_params={
                'numCores':1           
            }
        ) as qcgpj:
        
        print(f">> Executing on a HPC machine")
        campaign.execute(pool=qcgpj).collate()
else:

    campaign.execute().collate()

train_time=time.time() - start_time
print(f"> Finished training the models, time={train_time} s")

# Collate the results of all training runs
collation_results = campaign.get_collation_result()
pprint(collation_results)

# TODO Next: analysis, create a separate class to choose the best ML model
analysis = uq.analysis.BasicStats(qoi_cols=qoi)
campaign.apply_analysis(analysis)
results = campaign.get_last_analysis()

res_file = os.path.join(work_dir, f"hpo_res_{ft}.pickle")
with open(res_file, "bw") as rf:
    pickle.dump(results, rf)

#print(results)

analysis.analyse(collation_results)
analysis.analyse(results)

minrowidx = collation_results['loss'].idxmin()
print(f"Best model (for ft#{ft}) so far: {collation_results.iloc[minrowidx,:]}")

#TODO: get folder name of the current campaign and copy the right model file out

#loss = results.describe('loss')
#print(loss)
