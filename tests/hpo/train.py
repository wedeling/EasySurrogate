"""
Run a EasyVVUQ with a surrogate trainging as an application.
Varying parameters are the Hyperparameters of the surrogate models.
"""

import os
import easyvvuq as uq
import chaospy as cp
import pickle
import time
import numpy as np

import csv

# Ideally, here all the information on parameters should be accesses by sampler first
params = {
    "n_layers": {"type": "string", "min": 4, "max": 4, "default": 4},
    "n_neurons": {"type": "string", "min": 128, "max": 256, "default": 256}, 
} 
# TODO should be read from CSV; potentially: create a CSV from this script
# TODO force CSVSampler to interpret entries with correct type
vary = {}

campaign_name = 'hpo_easysurrogate_'
work_dir = '' #os.path.dirname('/hpo')

campaign = uq.Campaign(name=campaign_name, work_dir=work_dir)

# A file specifying a table of hyperparameter values (or their ranges/distributions) to pass for a number of ML models
# Mind: delimeter is ',' w/o spaces
param_file = 'hp_values.csv'

# Encoder should take a value from the sampler and pass it to EasySurrogate es.methos.*_Surrogate().train(...) as kwargs
encoder = uq.encoders.GenericEncoder(
    template_fname='hpo_template',
    delimiter='$',
    target_filename='input.json'
)

# Decoder should take a training, or better validation, loss values from an ML model for given hyperparameter value
decoder = uq.decoders.SimpleCSV(
    target_filename='output.json',
    output_columns=['RMSE']
)

# Execute should train a model in EasySurrogate: get the data, initalise object, call .train() and calculate the training/validation error
execute_train = uq.actions.ExecuteLocal(
    'python3 ../../single_model_train.py input.json &> train.log'
)

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

#for sample in sampler:
#    print(sample['n_layers'] == 3)

# Execute: train a number of ML models
print('> Starting to train the models')
start_time = time.time()

campaign.execute()

train_time=time.time() - start_time
print('> Finished training the models, time={}'.format(train_time))

# Next: analysis, create a separate class to choose the best ML model

