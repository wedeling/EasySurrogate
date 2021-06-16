"""
This script runs a simple MC EasyVVUQ Campign on the analytical Sobol g-function.
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

# EasyVUQ work directory
WORK_DIR = '/tmp'

# EasyVVUQ database location
ID = 'func'
DB_LOCATION = "sqlite:///" + WORK_DIR + "/campaign%s.db" % ID

########################
# EasyVVUQ MC Campaign #
########################

# choose a number of uncertain parameters (< 10)
D = 10

# Define parameter space
params = {}
for i in range(10):
    params["x%d" % (i + 1)] = {"type": "float",
                               "min": 0.0,
                               "max": 1.0,
                               "default": 0.5}
params["D"] = {"type": "integer", "default": D}
params["out_file"] = {"type": "string", "default": "output.csv"}
output_filename = params["out_file"]["default"]
output_columns = ["f"]

# the a vector determines the importance of each input
# a = np.linspace(0, np.sqrt(100), 10)**2
a = np.array([1 / 2 ** i for i in range(10)])
# a = np.zeros(10)
# a[0] = 1
# a[1] = 0.5
for i in range(10):
    params["a%d" % (i + 1)] = {"type": "float",
                               "min": 0.0,
                               "max": 100.0,
                               "default": a[i]}

# create encoder, decoder, and execute locally
encoder = uq.encoders.GenericEncoder(template_fname=HOME + '/model/func.template',
                                     delimiter='$',
                                     target_filename='in.json')
decoder = uq.decoders.SimpleCSV(target_filename=output_filename,
                                output_columns=output_columns)
execute = ExecuteLocal('{}/model/func.py in.json'.format(os.getcwd()))
actions = Actions(CreateRunDirectory(root=WORK_DIR),
                  Encode(encoder), execute, Decode(decoder))

# uncertain variables
vary = {}
for i in range(D):
    vary["x%d" % (i + 1)] = cp.Uniform(0, 1)

# Latin Hypercube sampler
my_sampler = uq.sampling.quasirandom.LHCSampler(vary=vary, max_num=1000)

# EasyVVUQ Campaign
campaign = uq.Campaign(name='func', params=params, actions=actions,
                       work_dir=WORK_DIR, db_location=DB_LOCATION)

# Associate the sampler with the campaign
campaign.set_sampler(my_sampler)

# Execute runs
campaign.execute().collate()