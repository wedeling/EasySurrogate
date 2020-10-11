from libmuscle.runner import run_simulation
from ymmsl import ComputeElement, Conduit, Configuration, Model, Settings

from micro import reduced_sgs
from macro import gray_scott_macro

import numpy as np
import json, sys

# the json input file containing the values of the parameters, and the
# output file
json_input = sys.argv[1]

with open(json_input, "r") as f:
    inputs = json.load(f)

#number of samples to run concurrently on a node
n_samples = 1

#get in n_samples input values from EasyVVUQ
feed = []; kill = []
for i in range(n_samples):
    feed.append(float(inputs['feed' + str(i)]))
    kill.append(float(inputs['kill' + str(i)]))
feed = np.array(feed)
kill = np.array(kill)

#Compute elements for the macro and micro model: specify the name of the python file here
elements = [ComputeElement('macro', 'macro', [n_samples]),
            ComputeElement('micro', 'micro', [n_samples])]

#connect the submodels
conduits = [Conduit('macro.state_out', 'micro.state'),
            Conduit('micro.sgs', 'macro.state_in')]

#create model instance
model = Model('gray_scott_reduced', elements, conduits)

#common settings
settings_dict = {'micro.t_max': 0.1, 'micro.dt': 0.1, 'N_Q': 2, 'N_LF': 128}

#parameter value settings, differs per run
for i in range(n_samples):
    settings_dict['macro[%d].feed' % i] = feed[i]
    settings_dict['macro[%d].kill' % i] = kill[i]
settings = Settings(settings_dict)

configuration = Configuration(model, settings)

#actual subroutines to run, imported from macro.p and micro.py
implementations = {'macro': gray_scott_macro, 'micro': reduced_sgs}

#execute ensemble on a single node
run_simulation(configuration, implementations)