import chaospy as cp
import numpy as np
import easyvvuq as uq
import os
# import fabsim3_cmd_api as fab
import matplotlib.pyplot as plt
import time

plt.close('all')

# author: Wouter Edeling
__license__ = "LGPL"

HOME = os.path.abspath(os.path.dirname(__file__))

# Set up a fresh campaign called "sc"
my_campaign = uq.Campaign(name='qmc', work_dir=HOME + '/EasyVVUQ_campaigns')

# number of uncertain parameters
d = 20

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

# Create an encoder, decoder and collation element
encoder = uq.encoders.GenericEncoder(
    template_fname=HOME + '/sc/poly.template',
    delimiter='$',
    target_filename='poly_in.json')
decoder = uq.decoders.SimpleCSV(target_filename=output_filename,
                                output_columns=output_columns)  # ,
# header=0)
# collater = uq.collate.AggregateSamples()

# Add the SC app (automatically set as current app)
my_campaign.add_app(name="sc",
                    params=params,
                    encoder=encoder,
                    decoder=decoder)  # ,
# collater=collater)

# uncertain variables
vary = {}
for i in range(d):
    vary["x%d" % (i + 1)] = cp.Uniform(0, 1)

# =================================
my_sampler = uq.sampling.MCSampler(vary=vary, n_mc_samples=100)

# Associate the sampler with the campaign
my_campaign.set_sampler(my_sampler)

# Will draw all (of the finite set of samples)
my_campaign.draw_samples()
t0 = time.time()
print(t0)
my_campaign.populate_runs_dir()
t1 = time.time()
print('Run time pupoulate run dirs = ', t1 - t0)

t0 = time.time()
print(t0)
# Use this instead to run the samples using EasyVVUQ on the localhost
my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(
    "./sc/poly_model.py poly_in.json"))
# fab.run_uq_ensemble(my_campaign.campaign_dir, script_name='poly_model',
#                     machine='localhost')
# fab.get_uq_samples(my_campaign.campaign_dir, machine='localhost')
t1 = time.time()
print('Run time sampling = ', t1 - t0)

t0 = time.time()
print(t0)
my_campaign.collate()
t1 = time.time()
print('Run time collate = ', t1 - t0)

data_frame = my_campaign.get_collation_result()

# Post-processing analysis
t0 = time.time()
print(t0)
analysis = uq.analysis.QMCAnalysis(sampler=my_sampler, qoi_cols=output_columns)
my_campaign.apply_analysis(analysis)
t1 = time.time()
print('Run time analysis = ', t1 - t0)

# some post-processing
results = my_campaign.get_last_analysis()

# analytic mean and standard deviation
# a = np.array([1/(2*(i+1)) for i in range(d)])
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

my_campaign.save_state('campaign2.json')
