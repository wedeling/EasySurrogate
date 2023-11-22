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
np.random.seed(42)

# Choice of flux tube - TODO make it a parameter / read from CSV
ft = sys.argv[1]

#TODO write down test fraction expliceitly - now it is 0.5 by default

# List all possible hyperparamters of a surrogate of this type, together with their types and default values
params = {
    "length_scale": {"type": "string", "min": 1e-12, "max": 1e+12, "default": "1.0"},
    "noize": {"type": "string", "min": 1e-16, "max": 1e+4, "default": "1e-3"}, 
    "bias": {"type": "string", "min": -1e+6, "max": 1e+6, "default": "0.0"},
    "nu_matern": {"type": "string", "min": 1e-6, "max": 1e+6, "default": "2.5"},
    "nu_stp": {"type": "string", "min": 2, "max": 1e+16, "default": "5"},
    "kernel": {"type": "string", "default": "Matern"},
    "testset_fraction": {"type": "string", "min": 0.0, "max": 1.0, "default": "0.5"},
    "n_iter" : {"type": "string", "min": 1, "default": "10"},
    "process_type" : {"type": "string", "default": "gaussian"},
    "backend" : {"type": "string", "default": "local"},
}
# looks like EasyVVUQ checks for a need in a default value after sampler is initialized 

# TODO should be read from CSV
# TODO force CSVSampler to interpret entries with correct type (int-s as int-s!)

# For Grid Search: form carthesian product of variables

vary = {} #TODO maybe: use vary to create CSV for non-categorical hyperparameters

# Define values for each parameter, create its cartesian grid, save as csv
param_search_vals = {
    "length_scale": [0.5, 1.0, 2.0],
    "noize": [1e-4, 1e-2, 1e-1], 
    "bias": [0., 1.0], # CURRENTLY not used by local implementation
    "nu_matern": [0.5, 1.5, 2.5], # not used by RBF and non-Matern kernels
    "nu_stp": [5, 10, 15], # not used for GPR (normal likelihood)
    "kernel": ['RBF', 'Matern'], 
    #"testset_fraction": [0.1, 0.5, 0.9], # does not reflect best model
    "n_iter" : [1, 5, 10], # CURRENTLY not used by local implementation
    "process_type" : ['gaussian', 'student_t'], # not used by scikit-learn implementation
    "backend" : ['scikit-learn'] # ['local', 'scikit-learn'],
}

csv_header = [k for k in param_search_vals.keys()]
csv_vals = [x for x in product(*[v for (k,v) in param_search_vals.items()])]

def clean_grid_by_rules(header, vals, def_vals):

    # Construct list of dictionaries of with keys taken from header and values taken from rows of vals
    data = [{header[i]:vals[j][i] for i in range(len(header))} for j in range(len(vals))]

    vals_new = []

    for d in data:
        if d['backend'] == 'local' and not m.isclose(d['bias'], float(def_vals['bias'])):
            continue
        if d['kernel'] != 'Matern' and not m.isclose(d['nu_matern'], float(def_vals['nu_matern'])):
            continue
        if d['process_type'] != 'student_t' and d['nu_stp'] != int(def_vals['nu_stp']):
            continue
        if d['backend'] == 'local' and d['n_iter'] != int(def_vals['n_iter']):
            continue
        if d['backend'] == 'scikit-learn' and d['process_type'] != def_vals['process_type']:
            continue

        vals_new.append([x for k,x in d.items()])

    print('> Using {0} different parameter values combinations instead of full {1}'.format(len(vals_new), len(vals)))

    return vals_new

csv_defaults = {k:v['default'] for k,v in params.items()}
csv_vals_new = clean_grid_by_rules(csv_header, csv_vals, csv_defaults)

with open('hp_values_gp_loc_2.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(csv_vals)

with open('hp_values_gp_loc_short_2.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(csv_vals_new)
 
# If run on HPC, should be called from the scheduler like SLURM
# for which an environmental variable HPC_EXECUTION should be specified
HPC_EXECUTION = os.environ['HPC_EXECUTION']

campaign_name = f"hpo_easysurrogate_f{ft}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
work_dir = ''
#TODO specify flexible paths

campaign = uq.Campaign(name=campaign_name, work_dir=work_dir)

# Optimising hyperparamters for GEM data
# Here listing different test cases for different combinations of varied parameters

# A file specifying a table of hyperparameter values (or their ranges/distributions) to pass for a number of ML models
# Mind: delimeter is ',' w/o spaces

#param_file = 'hp_values_gp.csv'
# For testset_frac=0.5 and kernel K=C(s0)*RBF(l1)+W(s1)+C(s2) best (default) parameters are l1=0.5, s1=0.001, s2=0.0
### 42n__z3x run_11 : 0.5 0.001 0.0
### vkbz8hz0 run_7: RBF l=0.5 s_n=.001 b=0. tfr=0.5 n_i=10, p=N be=loc

#param_file = 'hp_values_gp_tfrac.csv'
# For kernel parameters given above, a) lowering test dataset fraction to 0.5 gives ~0.33*|Err| decrease and platoeing
#                                    b) has a local minimum for test_frac=0.4
### _rdb0knh run_5 : 0.4

#param_file = 'hp_values_gp_niter.csv'
# For kernel parameters and test dataset fraction given above, the training gives best values for n_iter=10
### tp2csdpe run_6: 10; BUT single iteration could be enough

#param_file = 'hp_values_gp_stp.csv'
# Custom implementation: The relative test error of 0.048 for student_t process with Matern kernel, sigma_n=0.001, l=2.0
# w4mixs15 run_28 : Matern l=2. s_n=.001 b=0. tfr=0.5 n_i=10, p=STP be=scikit-learn
# tp2csdpe run_27 : Matern l=.5 s_n=.001 b=0. nu=1.5 tfr=0.5 n_i=10, p=STP be=scikit-learn

param_file = 'hp_values_gp_loc_short_2.csv'

# Encoder should take a value from the sampler and pass it to EasySurrogate es.methos.*_Surrogate().train(...) as kwargs
encoder = uq.encoders.GenericEncoder(
    template_fname='hpo_gp.template',
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
    f"python3 ../../../single_model_train_gp.py input.json {ft} > train.log"
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
print('> Starting to train the models')
start_time = time.time()

if HPC_EXECUTION:

    with QCGPJPool(
            #qcgpj_executor=QCGPJExecutor(),
            template=EasyVVUQParallelTemplate(),
            template_params={
                'numCores':1           
            }
        ) as qcgpj:
        
        print('>> Executing on a HPC machine')
        campaign.execute(pool=qcgpj).collate()
else:

    campaign.execute().collate()

train_time=time.time() - start_time
print('> Finished training the models, time={} s'.format(train_time))

# Collate the results of all training runs
collation_results = campaign.get_collation_result()
pprint(collation_results)

# TODO Next: analysis, create a separate class to choose the best ML model
analysis = uq.analysis.BasicStats(qoi_cols=qoi)
campaign.apply_analysis(analysis)
results = campaign.get_last_analysis()

res_file = os.path.join(work_dir, "hpo_res.pickle")
with open(res_file, "bw") as rf:
    pickle.dump(results, rf)

#print(results)

analysis.analyse(collation_results)
analysis.analyse(results)

minrowidx = collation_results['loss'].idxmin()
print("Best model so far: {0}".format(collation_results.iloc[minrowidx,:]))

#loss = results.describe('loss')
#print(loss)
