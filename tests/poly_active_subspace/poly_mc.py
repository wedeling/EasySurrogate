import chaospy as cp
import numpy as np
import easyvvuq as uq
import os
import matplotlib.pyplot as plt
import time
from easyvvuq.actions import CreateRunDirectory, Encode, Decode, CleanUp, ExecuteLocal, Actions

plt.close('all')

# author: Wouter Edeling
__license__ = "LGPL"

HOME = os.path.abspath(os.path.dirname(__file__))

# number of uncertain parameters
d = 5

# Define parameter space
params = {}
for i in range(d):
    params["x%d" % (i + 1)] = {"type": "float",
                               "min": 0.0,
                               "max": 1.0,
                               "default": 0.5}
params["d"] = {"type": "integer", "default": d}
params["out_file"] = {"type": "string", "default": "output.csv"}
output_filename = params["out_file"]["default"]
output_columns = ["f"]

encoder = uq.encoders.GenericEncoder(template_fname=HOME + '/sc/poly.template',
                                     delimiter='$', 
                                     target_filename='poly_in.json')
decoder = uq.decoders.SimpleCSV(target_filename=output_filename,
                                output_columns=output_columns)  
execute = ExecuteLocal('{}/sc/poly_model.py poly_in.json'.format(os.getcwd()))

actions = Actions(CreateRunDirectory('/tmp'), 
                  Encode(encoder), execute, Decode(decoder))

# uncertain variables
vary = {}
for i in range(d):
    vary["x%d" % (i + 1)] = cp.Uniform(0, 1)

my_sampler = uq.sampling.MCSampler(vary=vary, n_mc_samples=100)

campaign = uq.Campaign(name='mc', params=params, actions=actions)

# Associate the sampler with the campaign
campaign.set_sampler(my_sampler)

campaign.execute().collate()

data_frame = campaign.get_collation_result()

# Post-processing analysis
analysis = uq.analysis.QMCAnalysis(sampler=my_sampler, qoi_cols=output_columns)
campaign.apply_analysis(analysis)

# some post-processing
results = campaign.get_last_analysis()

# analytic mean and standard deviation
a = np.zeros(d)
a[0] = 1
a[1] = 1.0
ref_mean = np.prod(a + 1) / 2**d
ref_std = np.sqrt(np.prod(9 * a[0:d]**2 / 5 + 2 * a[0:d] + 1) / 2**(2 * d) - ref_mean**2)

print("======================================")
print("Number of samples = %d" % my_sampler.n_samples())
print("--------------------------------------")
print("Analytic mean = %.4e" % ref_mean)
print("Computed mean = %.4e" % results.describe('f', 'mean'))
print("--------------------------------------")
print("Analytic standard deviation = %.4e" % ref_std)
print("Computed standard deviation = %.4e" % results.describe('f', 'std'))
print("--------------------------------------")
