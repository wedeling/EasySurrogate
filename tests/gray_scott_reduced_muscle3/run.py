from libmuscle.runner import run_simulation
from ymmsl import ComputeElement, Conduit, Configuration, Model, Settings

# load the macro and micro model
from micro import reduced_sgs
from macro import gray_scott_macro

# Compute elements for the macro and micro model: specify the name of the python file here
elements = [ComputeElement('macro', 'macro', [1]),
            ComputeElement('micro', 'micro', [1])]

# connect the submodels
conduits = [Conduit('macro.state_out', 'micro.state_in'),
            Conduit('micro.sgs_out', 'macro.sgs_in')]

# create model instance
model = Model('gray_scott_reduced', elements, conduits)

# settings
settings = Settings({'micro.t_max': 0.5, 'micro.dt': 0.5, 'N_Q': 2,
                     'N_LF': 128,
                     'macro[0].feed': 0.02,
                     'macro[0].kill': 0.05,
                     })
# configure the model
configuration = Configuration(model, settings)

# actual subroutines to run, imported from macro.py and micro.py
implementations = {'macro': gray_scott_macro, 'micro': reduced_sgs}

# run the coupled model
run_simulation(configuration, implementations)
