"""
Run a EasyVVUQ with a surrogate trainging as an application.
Varying parameters are the Hyperparameters of the surrogate models.
"""

import os
import easyvvuq as uq
import pickle
import time

#TODO do the same for GPR: 
# parameters like noize_level and Matern lengthscale in CSV, pass in script of the highest level
# train on GEM data
# input HPs and output validation R2 and RMSE JSON for training script

# Ideally, here all the information on parameters should be accesses by sampler first
params = {
    "length_scale": {"type": "string", "min": 1e-6, "max": 1e+6, "default": 1.0},
    "noize": {"type": "string", "min": 1e-16, "max": 1e+3, "default": 1e-8}, 
    "bias":{"type": "string", "min": -1e+4, "max": 1e+4, "default": 1.0},
} 
# TODO should be read from CSV; potentially: create a CSV from this script
# TODO force CSVSampler to interpret entries with correct type
vary = {}

campaign_name = 'hpo_easysurrogate_'
work_dir = '' #os.path.dirname('/hpo')

campaign = uq.Campaign(name=campaign_name, work_dir=work_dir)

# A file specifying a table of hyperparameter values (or their ranges/distributions) to pass for a number of ML models
# Mind: delimeter is ',' w/o spaces
param_file = 'hp_values_gp.csv'

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

# Sampler should read hyperparameter-value dictionary from a CSV on a harddrive
sampler = uq.sampling.CSVSampler(filename=param_file) 
# TODO: sampler has to read numbers as integers
campaign.set_sampler(sampler)

# Execute: train a number of ML models
print('> Starting to train the models')
start_time = time.time()

campaign.execute().collate()

train_time=time.time() - start_time
print('> Finished training the models, time={}'.format(train_time))

# Collate the results of all training runs
collation_results = campaign.get_collation_result()
print(collation_results)

# TODO Next: analysis, create a separate class to choose the best ML model
analysis = uq.analysis.BasicStats(qoi_cols=qoi)
campaign.apply_analysis(analysis)
results = campaign.get_last_analysis()


res_file = os.path.join(work_dir, "hpo_res.pickle")
with open(res_file, "bw") as rf:
    pickle.dump(results, rf)

print(results)
print(collation_results.min('test_error'))

# TODO check if error is read as a string
#test_error = results.describe('test_error')
#print(test_error)
