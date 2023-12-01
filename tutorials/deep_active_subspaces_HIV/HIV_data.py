import numpy as np
import matplotlib.pyplot as plt
from HIV_model import *
import easysurrogate as es

plt.close('all')

#Number of random points to use
N = 1000

# Number of inputs
D = 27

#Nominal Parameter Values
nominal = np.array([10, .15, 5, .2, 55.6, 3.87e-3, 1e-6, 4.5e-4, 7.45e-4, 5.22e-4, 3e-6,\
    3.3e-4, 6e-9, .537, .285, 7.79e-6, 1e-6, 4e-5, .01, .28, .05, .005, .005, .015, 2.39,\
    3e-4, .97])

#Lower and upper parameter limits
xl = .975*nominal; xu = 1.025*nominal

#Normalized parameter values
p = np.random.uniform(-1, 1, (N, len(xl)))
times = np.array([5, 15, 24, 38, 40, 45, 50, 55, 65, 90, 140, 500, 750,
                  1000, 1600, 1800, 2000, 2200, 2400, 2800, 3400])
T = times.size

# derivatives = np.zeros([N, T, D])
derivatives_norm = np.zeros([N, D])

#Step sizes to use for finite differences
h = 1e-7

# sample Tcells N times until each specified time, with a time step of 1
f0 = Tcells(p, np.linspace(1, times[-1], times[-1]))
samples = f0[:, times-1]
mu = np.mean(samples, axis=0)
sigma = np.std(samples, axis=0)

samples_perturbed = np.zeros([N, T, D])
# compute the derivatives of (the norm of) f for all D inputs using FD
for j in range(D):
    print(j)
    p1 = np.copy(p)
    p1[:, j] += h
    f1 = Tcells(p1, np.linspace(1, times[-1], times[-1]))
    samples_perturbed[:, :, j] = f1[:, times-1]

    # derivative of the standardized norm, wrt the normalized inputs in [-1, 1]
    derivatives_norm[:, j] = (np.linalg.norm((samples_perturbed[:,:,j] - mu) / sigma, axis=1) - 
                              np.linalg.norm((samples - mu) / sigma, axis=1)) / h

# store data
campaign = es.Campaign()
campaign.store_data_to_hdf5({'inputs':p, 'outputs':samples, 'derivatives_norm':derivatives_norm},
                            file_path='my_samples_w_norm_deriv.hdf5')
