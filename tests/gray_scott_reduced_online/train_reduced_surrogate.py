"""
A 2D Gray-Scott reaction diffusion model, with reduced sungrid scale surrogate

Numerical method:
Craster & Sassi, spectral algorithmns for reaction-diffusion equations, 2006.

Reduced SGS:
W. Edeling, D. Crommelin, Reducing data-driven dynamical subgrid scale
models by physical constraints, Computer & Fluids, 2020.

This script runs the Gray-Scott model at lower resolution, and uses the
training data of the statistics of interest to drive a reduced SGS term.
"""

import numpy as np
import easysurrogate as es
import time
import h5py
import tkinter as tk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
import GrayScott_2D as gs

def draw():
    """
    simple plotting routine
    """
    plt.clf()

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # ct = ax1.contourf(xx, yy, u, 100)
    # ct = ax2.contourf(xx, yy, v, 100)

    ax1.plot(T, plot_dict_HF[0], label='model')
    ax1.plot(T, plot_dict_ref[0], 'o', label='ref')
    ax1.legend(loc=0)
    ax2.plot(T, plot_dict_HF[1], label='model')
    ax2.plot(T, plot_dict_ref[1], 'o', label='ref')
    ax2.legend(loc=0)
    ax3.plot(T, plot_dict_HF[2], label='model')
    ax3.plot(T, plot_dict_ref[2], 'o', label='ref')
    ax3.legend(loc=0)
    ax4.plot(T, plot_dict_HF[3], label='model')
    ax4.plot(T, plot_dict_ref[3], 'o', label='ref')
    ax4.legend(loc=0)

    plt.tight_layout()

    plt.pause(0.1)

def compute_int(X1_hat, X2_hat, N):
    """
    Compute the integral of X1*X2 using the Fourier expansion
    """
    integral = np.dot(X1_hat.flatten(), np.conjugate(X2_hat.flatten())) / N**4
    return integral.real


plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'
HOME = os.path.abspath(os.path.dirname(__file__))

##############################
# Easysurrogate modification #
##############################

# create campaign
campaign = es.Campaign()

# load reference data of the statistics of interest
data_frame = campaign.load_hdf5_data(file_path=HOME + '/samples/gray_scott_reference.hdf5')
Q_ref = data_frame['Q_HF']

# lower the number of gridpoints in 1D compared to ref data
I = 7
N = 2**I

# number of stats to track per PDE  (we have 2 PDEs)
N_Q = 2

# reduced basis function for the average concentrations of u and v
V_hat_1 = np.fft.fft2(np.ones([N, N]))

# create a reduced SGS surrogate object
surrogate = es.methods.Reduced_Surrogate(N_Q, N)

# add surrogate to campaign
campaign.add_app(name="gray_scott_reduced", surrogate=surrogate)

##################################
# End Easysurrogate modification #
##################################

# domain size [-L, L]
L = 1.25

# user flags
plot = True
store = True
state_store = True
restart = True

sim_ID = 'test_gray_scott_reduced'

if plot:
    fig = plt.figure(figsize=[8, 8])
    plot_dict_HF = {}
    plot_dict_ref = {}
    T = []
    for i in range(2 * N_Q):
        plot_dict_HF[i] = []
        plot_dict_ref[i] = []

# TRAINING DATA SET
QoI = ['Q_LR']
Q = len(QoI)

# allocate memory
samples = {}

if store:
    samples['N'] = N

    for q in range(Q):
        samples[QoI[q]] = []

# diffusion coefficients
epsilon_u = 2e-5
epsilon_v = 1e-5

# alpha pattern
feed = 0.02
kill = 0.05

# beta pattern
# feed = 0.02
# kill = 0.045

# epsilon pattern
# feed = 0.02
# kill = 0.055

# time step parameters
dt = 0.1
n_steps = 1000
plot_frame_rate = 100
store_frame_rate = 1
t = 0.0

gray_scott_LR = gs.GrayScott_2D(N, L, dt, feed, kill, epsilon_u, epsilon_v)

# Initial condition
if restart:
    u_hat, v_hat = gray_scott_LR.load_state('./restart/state_LR.pickle')
else:
    u_hat, v_hat = gray_scott_LR.initial_cond()

# # Integrating factors
# int_fac_u, int_fac_u2, int_fac_v, int_fac_v2 = integrating_factors(k_squared)

# counters
j = 0
j2 = 0

V_hat_1 = np.fft.fft2(np.ones([N, N]))

t0 = time.time()
# time stepping
for n in range(n_steps):

    if np.mod(n, 1000) == 0:
        print('time step %d of %d' % (n, n_steps))

    ##############################
    # Easysurrogate modification #
    ##############################

    # compute reference stats
    Q_LR = np.zeros(2 * N_Q)
    Q_LR[0] = compute_int(V_hat_1, u_hat, N)
    Q_LR[1] = 0.5 * compute_int(u_hat, u_hat, N)
    Q_LR[2] = compute_int(V_hat_1, v_hat, N)
    Q_LR[3] = 0.5 * compute_int(v_hat, v_hat, N)

    # train the two reduced sgs source terms using the recieved reference data Q_ref
    reduced_dict_u = surrogate.train([V_hat_1, u_hat], Q_ref[n][0:N_Q], Q_LR[0:N_Q])
    reduced_dict_v = surrogate.train([V_hat_1, v_hat], Q_ref[n][N_Q:], Q_LR[N_Q:])
    
    # get the two reduced sgs terms from the dict
    reduced_sgs_u = np.fft.ifft2(reduced_dict_u['sgs_hat'])
    reduced_sgs_v = np.fft.ifft2(reduced_dict_v['sgs_hat'])

    # pass reduced sgs terms to rk4
    u_hat, v_hat = gray_scott_LR.rk4(u_hat, v_hat, 
                                     reduced_sgs_u=reduced_sgs_u, reduced_sgs_v=reduced_sgs_v)
    
    # we do not want to store the full field reduced source terms
    del reduced_dict_u['sgs_hat']
    del reduced_dict_v['sgs_hat']

    # accumulate the time series in the campaign object
    campaign.accumulate_data(reduced_dict_u, names=['c_ij_u',
                                                    'inner_prods_u', 'src_Q_u',
                                                    'tau_u'])

    campaign.accumulate_data(reduced_dict_v, names=['c_ij_v',
                                                    'inner_prods_v', 'src_Q_v',
                                                    'tau_v'])

    ##################################
    # End Easysurrogate modification #
    ##################################

    j += 1
    j2 += 1
    t += dt
    # plot while running simulation
    if j == plot_frame_rate and plot:
        j = 0
        u = np.fft.ifft2(u_hat)
        v = np.fft.ifft2(v_hat)

        for i in range(2 * N_Q):
            plot_dict_HF[i].append(Q_LR[i])
            plot_dict_ref[i].append(Q_ref[n][i])

        T.append(t)

        draw()

    if j2 == store_frame_rate and store:
        j2 = 0

        for qoi in QoI:
            samples[qoi].append(eval(qoi))

t1 = time.time()
print('*************************************')
print('Simulation time = %f [s]' % (t1 - t0))
print('*************************************')

# store the state of the system to allow for a simulation restart at t > 0
if state_store:
    gray_scott_LR.store_state('./restart/state_LR.pickle')