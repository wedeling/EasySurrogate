"""
This script runs a simple MC EasyVVUQ Campign on the analytical Sobol g-function.
The EasyVVUQ data frame is then read by EasySurrogate, and a neural network is trained
on the input-output data geerated by EasyVVUQ.
"""

import os
import easysurrogate as es
import chaospy as cp
import numpy as np
import easyvvuq as uq
import matplotlib.pyplot as plt
from easyvvuq.actions import CreateRunDirectory, Encode, Decode, ExecuteLocal, Actions

plt.close('all')

# author: Wouter Edeling
__license__ = "LGPL"

# the absolute path of this file
HOME = os.path.abspath(os.path.dirname(__file__))

########################
# EasyVVUQ MC Campaign #
########################

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
my_sampler = uq.sampling.MCSampler(vary=vary, n_mc_samples=100)

# EasyVVUQ Campaign
campaign = uq.Campaign(name='g_func', params=params, actions=actions)

# Associate the sampler with the campaign
campaign.set_sampler(my_sampler)

# Execute runs
campaign.execute().collate()

# get the EasyVVUQ data frame
data_frame = campaign.get_collation_result()

# # Post-processing analysis
# analysis = uq.analysis.QMCAnalysis(sampler=my_sampler, qoi_cols=output_columns)
# campaign.apply_analysis(analysis)

# # some post-processing
# results = campaign.get_last_analysis()

# # analytic mean and standard deviation
# a = np.zeros(D)
# a[0] = 1
# a[1] = 1.0
# ref_mean = np.prod(a + 1) / 2**D
# ref_std = np.sqrt(np.prod(9 * a[0:D]**2 / 5 + 2 * a[0:D] + 1) / 2**(2 * D) - ref_mean**2)

# print("======================================")
# print("Number of samples = %d" % my_sampler.n_samples())
# print("--------------------------------------")
# print("Analytic mean = %.4e" % ref_mean)
# print("Computed mean = %.4e" % results.describe('f', 'mean'))
# print("--------------------------------------")
# print("Analytic standard deviation = %.4e" % ref_std)
# print("Computed standard deviation = %.4e" % results.describe('f', 'std'))
# print("--------------------------------------")

##############################
# EasySurrogate ANN campaign #
##############################

# Create an EasySurrogate campaign
surr_campaign = es.Campaign()

# This is the main point of this test: extract training data from EasyVVUQ data frame
features, samples = surr_campaign.load_easyvvuq_data(campaign, qoi_cols='f')

# Create artificial neural network surrogate
surrogate = es.methods.ANN_Surrogate()

# Number of training iterations (number of mini batches)
N_ITER = 10000

# The latter fraction of the data to be kept apart for testing
TEST_FRAC = 0.3

# Train the ANN
surrogate.train(features, samples['f'], N_ITER,
                n_layers=4, n_neurons=50, test_frac=TEST_FRAC)

# get some useful dimensions of the ANN surrogate
dims = surrogate.get_dimensions()

# evaluate the ANN surrogate on the training data
training_predictions = np.zeros([dims['n_train'], dims['n_out']])
for i in range(dims['n_train']):
    training_predictions[i] = surrogate.predict(features[i])

# print the relative training error
error_train = np.linalg.norm(training_predictions - samples['f'][0:dims['n_train']]) /\
    np.linalg.norm(samples['f'][0:dims['n_train']])
print("Relative error on training set = %.3f percent" % (error_train * 100))

# evaluate the ANN surrogate on the test data
test_predictions = np.zeros([dims['n_test'], dims['n_out']])
for count, i in enumerate(range(dims['n_train'], dims['n_samples'])):
    test_predictions[count] = surrogate.predict(features[i])

# print the relative test error
error_test = np.linalg.norm(test_predictions - samples['f'][dims['n_train']:]) /\
    np.linalg.norm(samples['f'][dims['n_train']:])
print("Relative error on test set = %.3f percent" % (error_test * 100))
