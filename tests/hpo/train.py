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

# Ideally, here all the information on parameters should be accesses by sampler first
params = {}
vary = {}

campaign_name = 'hpo_easysurrogate'
work_dir = os.path.dirname('/hpo')

campaign = uq.Campaign(name=campaign_name, work_dir=work_dir)

# A file specifying a table of hyperparameter values (or their ranges/distributions) to pass for a number of ML models
param_file = 'hp_values.csv'

# Encoder should take a value from the sampler and pass it to EasySurrogate es.methos.*_Surrogate().train(...) as kwargs
encoder = uq.encoders.GenericEncoder(
    target_filename='input.json'
)

# Decoder should take a training, or better validation, loss values from an ML model for given hyperparameter value
decoder = uq.decoders.SimpleCSV()

# Execute should train a model in EasySurrogate: get the data, initalise object, call .train() and calculate the training/validation error
execute_train = uq.actions.ExecuteLocal(
    'python3 single_model_train.py input.json'
)

actions = uq.actions.Actions(
    #uq.actions.CreateRunDirectory('/runs'),
    uq.actions.Encode(encoder),
    execute_train,
    uq.actions.Decode(decoder),
)

campaign.add_app(
    name=campaign_name,
    actions=actions,
)

# Sampler should read hyperparameter-value dictionary from a CSV on a harddrive
sampler = uq.sampling.CSVSampler(filename=param_file)
campaign.set_sampler(sampler)

# Execute: train a number of ML models
campaign.execute()

# Next: analysis, create a separate class to choose the best ML model

