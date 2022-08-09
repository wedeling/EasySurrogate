"""
Run a EasyVVUQ with a surrogate trainging as an application.
Varying parameters are the Hyperparameters of the surrogate models.
"""

import os
import pickle
import time

import easyvvuq as uq
from easyvvuq.actions import QCGPJPool
from easyvvuq.actions.execute_qcgpj import EasyVVUQParallelTemplate

from qcg.pilotjob.executor_api.qcgpj_executor import QCGPJExecutor

from pprint import pprint
from itertools import product
import csv

import easysurrogate as es

#TODO: ADD A RANDOM SEED

# List all possible hyperparamters of a surrogate of this type, together with their types and default values
params = {
    "length_scale": {"type": "string", "min": 1e-6, "max": 1e+6, "default": "1.0"},
    "noize": {"type": "string", "min": 1e-16, "max": 1e+3, "default": "1e-8"}, 
    "bias": {"type": "string", "min": -1e+4, "max": 1e+4, "default": "0.0"},
    "nu": {"type": "string", "min": 1e-4, "max": 1e+4, "default": "2.5"},
    "kernel": {"type": "string", "default": "Matern"},
    "testset_fraction": {"type": "string", "min": 0.0, "max": 1.0, "default": "0.5"},
    "n_iter" : {"type": "string", "min": 1, "default": "10"},
    "process_type" : {"type": "string", "default": "gaussian"},
    "backend" : {"type": "string", "default": "local"},
}
# looks like EasyVVUQ checks for a need in a default value after sampler is initialized 

# TODO should be read from CSV; potentially: create a CSV from this script
# TODO force CSVSampler to interpret entries with correct type

# For Grid Search: form carthesian product of variables

vary = {} #TODO maybe: use vary to create CSV for non-categorical hyperparameters

# Defien values for each parameter, create its cartesian grid, save as csv
param_search_vals = {
    "length_scale": [0.5, 1.0, 2.0],
    "noize": [1e-4, 1e-2, 1e-1], 
    "bias": [0., 1.0],
    "nu": [0.5, 1.5, 2.5], # TODO: REFACTOR FOR UNAMBIGIOUS DIFFERENTIATION OF STUDENT-T NU AND MATERN NU
    "kernel": ['RBF', 'Matern'],
    "testset_fraction": [0.1, 0.5, 0.9],
    "n_iter" : [1, 5, 10],
    "process_type" : ['gaussian', 'student_t'],
    "backend" : ['local', 'scikit-learn'],
}

csv_header = [k for k in param_search_vals.keys()]
csv_vals = [x for x in product(*[v for (k,v) in param_search_vals.items()])]

with open('hp_values_gp_loc.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(csv_vals)

#TODO: IMPORTANT: when runnign for the large csv there is a failure due to not positive definite covariance matrix
#  when training a Student-t (nu=1.5) model with local implementation, Matern kernel: run_143 !!!
 
# If run on HPC, should be called from the scheduler like SLURM
# for which an environmental variable HPC_EXECUTION should be specified
HPC_EXECUTION = os.environ['HPC_EXECUTION']

campaign_name = 'hpo_easysurrogate_'
work_dir = ''
#TODO specify flexible paths

campaign = uq.Campaign(name=campaign_name, work_dir=work_dir)

# Optimising hyperparamters for GEM data 

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

param_file = 'hp_values_gp_stp.csv'
# Custom implementation: The relative test error of 0.048 for student_t process with Matern kernel, sigma_n=0.001, l=2.0
# w4mixs15 run_28 : Matern l=2. s_n=.001 b=0. tfr=0.5 n_i=10, p=STP be=scikit-learn
# tp2csdpe run_27 : Matern l=.5 s_n=.001 b=0. nu=1.5 tfr=0.5 n_i=10, p=STP be=scikit-learn

#param_file = 'hp_values_gp_loc.csv'

# Encoder should take a value from the sampler and pass it to EasySurrogate es.methos.*_Surrogate().train(...) as kwargs
encoder = uq.encoders.GenericEncoder(
    template_fname='hpo_gp.template',
    delimiter='$',
    target_filename='input.json'
)

# Decoder should take a training, or better validation, loss values from an ML model for given hyperparameter value
qoi = ['test_error']
decoder = uq.decoders.JSONDecoder(
    target_filename='output.json',
    output_columns=qoi,
)

# Execute should train a model in EasySurrogate: get the data, initalise object, call .train() and calculate the training/validation error
execute_train = uq.actions.ExecuteLocal(
    'python3 ../../../single_model_train_gp.py input.json > train.log'
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

minrowidx = collation_results['test_error'].idxmin()
print("Best model so far: {0}".format(collation_results.iloc[minrowidx,:]))

# TODO check if error is read as a string
#test_error = results.describe('test_error')
#print(test_error)
