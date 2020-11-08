"""
===============================================================================
Generate Lorenz 96 data for machine learning
===============================================================================
"""

####################################
# Lorenz 96 subroutines (2 Layers) #
####################################

import numpy as np
import easysurrogate as es
from matplotlib import animation
import matplotlib.pyplot as plt


def rhs_X(X, B, **kwargs):
    """
    Compute the right-hand side of X

    Parameters:
        - X (array, size K): large scale variables
        - B (array, size K): SGS term

    returns:
        - rhs_X (array, size K): right-hand side of X
    """
    
    if 'nudge' in kwargs:
        nudge = kwargs['nudge']
    else:
        nudge = 0.0

    rhs_X = np.zeros(K)

    # first treat boundary cases (k=1, k=2 and k=K)
    rhs_X[0] = -X[K - 2] * X[K - 1] + X[K - 1] * X[1] - X[0] + F

    rhs_X[1] = -X[K - 1] * X[0] + X[0] * X[2] - X[1] + F

    rhs_X[K - 1] = -X[K - 3] * X[K - 2] + X[K - 2] * X[0] - X[K - 1] + F

    # treat interior points
    for k in range(2, K - 1):
        rhs_X[k] = -X[k - 2] * X[k - 1] + X[k - 1] * X[k + 1] - X[k] + F

    rhs_X += B + nudge

    return rhs_X


def rhs_Y_k(Y, X_k, k):
    """
    Compute the right-hand side of Y for fixed k

    Parameters:
        - Y (array, size (J,K)): small scale variables
        - X_k (float): large scale variable X_k
        - k (int): index k

    returns:
        - rhs_Yk (array, size J): right-hand side of Y for fixed k
    """

    rhs_Yk = np.zeros(J)

    # first treat boundary cases (j=1, j=J-1, j=J-2)
    if k > 0:
        idx = k - 1
    else:
        idx = K - 1
    rhs_Yk[0] = (Y[1, k] * (Y[J - 1, idx] - Y[2, k]) - Y[0, k] + h_y * X_k) / epsilon

    if k < K - 1:
        idx = k + 1
    else:
        idx = 0
    rhs_Yk[J - 2] = (Y[J - 1, k] * (Y[J - 3, k] - Y[0, idx]) - Y[J - 2, k] + h_y * X_k) / epsilon
    rhs_Yk[J - 1] = (Y[0, idx] * (Y[J - 2, k] - Y[1, idx]) - Y[J - 1, k] + h_y * X_k) / epsilon

    # treat interior points
    for j in range(1, J - 2):
        rhs_Yk[j] = (Y[j + 1, k] * (Y[j - 1, k] - Y[j + 2, k]) - Y[j, k] + h_y * X_k) / epsilon

    return rhs_Yk


def step_X(X_n, f_nm1, B, dt, **kwargs):
    """
    Time step for X equation, using Adams-Bashforth

    Parameters:
        - X_n (array, size K): large scale variables at time n
        - f_nm1 (array, size K): right-hand side of X at time n-1
        - B (array, size K): SGS term

    Returns:
        - X_np1 (array, size K): large scale variables at time n+1
        - f_nm1 (array, size K): right-hand side of X at time n
    """

    # right-hand side at time n
    f_n = rhs_X(X_n, B, **kwargs)

    # adams bashforth
    X_np1 = X_n + dt * (1.5 * f_n - 0.5 * f_nm1)

    return X_np1, f_n


def step_Y(Y_n, g_nm1, X_n, dt):
    """
    Time step for Y equation, using Adams-Bashforth

    Parameters:
        - Y_n (array, size (J,K)): small scale variables at time n
        - g_nm1 (array, size (J, K)): right-hand side of Y at time n-1
        - X_n (array, size K): large scale variables at time n

    Returns:
        - Y_np1 (array, size (J,K)): small scale variables at time n+1
        - g_nm1 (array, size (J,K)): right-hand side of Y at time n
    """

    g_n = np.zeros([J, K])
    for k in range(K):
        g_n[:, k] = rhs_Y_k(Y_n, X_n[k], k)

    multistep_rhs = dt * (1.5 * g_n - 0.5 * g_nm1)

    Y_np1 = Y_n + multistep_rhs

    return Y_np1, g_n

###############
# Main program
###############

plt.close('all')

#####################
# Lorenz96 parameters
#####################

# K = 36
# J = 10
# F = 10.0
# h_x = -1.0
# h_y = 1.0
# epsilon = 0.1

K = 36
J = 10
F = 7.0
h_x = -2.0
h_y = 2.0
epsilon = 0.2

##################
# Time parameters
##################

dt_HR = 0.001
N = 10
dt_LR = N*dt_HR
tau_nudge = dt_LR
t_end = 50.0
t = np.arange(0.0, t_end, dt_LR)
M = 10

###################
# Simulation flags
###################

make_movie = False  # make a movie
store = False  # store the prediction results

#############################
# Original initial condition
#############################

# # equilibrium initial condition for X, zero IC for Y
# X_n = np.ones(K) * F
# X_n[10] += 0.01  # add small perturbation to 10th variable

# # initial condition small-scale variables
# Y_n = np.zeros([J, K])
# B_n = h_x * np.mean(Y_n, axis=0)

# # initial right-hand sides
# f_nm1 = rhs_X(X_n, B_n)

# g_nm1 = np.zeros([J, K])
# for k in range(K):
#     g_nm1[:, k] = rhs_Y_k(Y_n, X_n[k], k)

##############################
# Easysurrogate modification #
##############################

# load pre-trained campaign
campaign = es.Campaign(load_state=True)

# set the batch size equal to M
campaign.surrogate.surrogate.set_batch_size(1)

# change IC
data_frame = campaign.load_hdf5_data()
X_n = data_frame['X_data'][campaign.surrogate.max_lag]
Y_n = data_frame['Y_data'][campaign.surrogate.max_lag]
B_n = data_frame['B_data'][campaign.surrogate.max_lag]

# initial right-hand side
f_nm1 = rhs_X(X_n, B_n)
    
#low-res variable init
X_n_LR = X_n
B_n_LR = np.zeros(K)
f_nm1_LR = rhs_X(X_n_LR, B_n_LR)

g_nm1 = np.zeros([J, K])
for k in range(K):
    g_nm1[:, k] = rhs_Y_k(Y_n, X_n[k], k)

##################################
# End Easysurrogate modification #
##################################

# allocate memory for solutions
X_data = np.zeros([t.size, K])
Y_data = np.zeros([t.size, J, K])
B_data = np.zeros([t.size, K])

X_data_LR = np.zeros([t.size, K])

#allocate feature and target memory
features = []; targets = []

# start time integration
idx = 0
for t_i in t:
    
    #store Delta X
    Delta_X = X_n_LR - X_n
    
    #Store LR and HR state at beginning of time step
    X_n_LR_hat = X_n_LR
    X_n_hat = X_n
    
    features.append(X_n_LR_hat)

    for n in range(N):
        # solve small-scale equation
        Y_np1, g_n = step_Y(Y_n, g_nm1, X_n, dt_HR)
        # compute SGS term
        B_n = h_x*np.mean(Y_n, axis=0)

        # solve large-scale equation
        X_np1, f_n = step_X(X_n, f_nm1, B_n, dt_HR, nudge = Delta_X / tau_nudge)

        # update variables
        X_n = np.copy(X_np1)
        Y_n = np.copy(Y_np1)
        f_nm1 = np.copy(f_n)
        g_nm1 = np.copy(g_n)

    #Compute "assumed" HR increment
    Delta_X_internal_HR = X_n - X_n_hat - Delta_X / tau_nudge * dt_LR
   
    # solve LR equation
    X_np1_LR, f_n_LR = step_X(X_n_LR, f_nm1_LR, B_n_LR, dt_LR)

    # compute the target for the neural network
    targets.append((Delta_X_internal_HR - (X_np1 - X_n_hat)) / dt_LR)

    # evaluate the neural network
    B = []
    for k in range(K):
        B.append(campaign.surrogate.predict(X_n_LR_hat[k]))
    B = np.array(B).flatten()
    
    # update the LR state
    X_np1_LR += B * dt_LR
    
    # update the neural network every M time steps
    if np.mod(idx, M) == 0 and idx != 0:
        features = np.array(features).flatten()
        targets = np.array(targets).reshape([-1, 1])
        campaign.surrogate.train([features], targets, [[1]], 500, n_layers=3, n_neurons=32,
                                 batch_size=1)
        features = []; targets = []        

    #update low-res vars
    X_n_LR = np.copy(X_np1_LR)
    f_nm1_LR = np.copy(f_n_LR)

    # store solutions
    X_data[idx, :] = X_np1
    # Y_data[idx, :] = Y_np1
    B_data[idx, :] = B_n
    X_data_LR[idx, :] = X_np1_LR
    idx += 1

    if np.mod(idx, 1000) == 0:
        print('t =', np.around(t_i, 1), 'of', t_end)

if store:
    campaign.store_data_to_hdf5({'X_data': X_data, 'B_data': B_data})

# plot results
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
theta = np.linspace(0.0, 2.0 * np.pi, K + 1)
# create a while in the middle to the plot, scales plot better
ax.set_rorigin(-22)
# remove set radial tickmarks, no labels
ax.set_rgrids([-10, 0, 10], labels=['', '', ''])[0][1]
ax.legend(loc=1)
ax.plot(theta, np.append(X_data[-1, :], X_data[-1, 0]), label='X')
ax.plot(theta, np.append(B_data[-1, :], B_data[-1, 0]), label='B')
plt.show()

#############
# Plot PDEs #
#############

post_proc = es.analysis.BaseAnalysis()

fig = plt.figure()
ax = fig.add_subplot(111, xlabel=r'$X_k$')
X_dom, X_pde = post_proc.get_pdf(X_data.flatten()[0:-1:10])
X_dom_LR, X_pde_LR = post_proc.get_pdf(X_data_LR.flatten()[0:-1:10])
ax.plot(X_dom, X_pde, 'ko', label='HR')
ax.plot(X_dom_LR, X_pde_LR, 'b', label='LR')
plt.yticks([])
plt.legend(loc=0)
plt.tight_layout()

plt.show()

