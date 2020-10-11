from libmuscle.runner import run_simulation
from ymmsl import ComputeElement, Conduit, Configuration, Model, Settings

from micro import reduced_sgs
from macro import gray_scott_macro

elements = [ComputeElement('macro', 'macro', [1]),
            ComputeElement('micro', 'micro', [1])]

conduits = [Conduit('macro.state_out', 'micro.state'),
            Conduit('micro.sgs', 'macro.state_in')]

model = Model('gray_scott_reduced', elements, conduits)

settings = Settings({'micro.t_max': 0.5, 'micro.dt': 0.5, 'N_Q': 2, 
                     'N_LF': 128, 
                     'macro[0].feed': 0.02, 
                     'macro[0].kill': 0.05,
                     })

configuration = Configuration(model, settings)

implementations = {'macro': gray_scott_macro, 'micro': reduced_sgs}

run_simulation(configuration, implementations)