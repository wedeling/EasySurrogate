"""
===============================================================================
This file recreates the Lorenz96 results from:

     Rasp, "Coupled online learning as a way to tackle instabilities
     and biases in neural network parameterizations:
     general algorithms and Lorenz 96 case study", 2020.

    The neural net is a standard ANN that is trained and applied locally,
    with Markovian features

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

# predict under 'difficult' parameter settings, with neural trained under
# 'easy' settings

# K = 36
# J = 10
# F = 10.0
# h_x = -1.0
# h_y = 1.0
# epsilon = 0.1

# #The other way around

# K = 36
# J = 10
# F = 7.0
# h_x = -2.0
# h_y = 2.0
# epsilon = 0.2

K = 18
J = 20
F = 10.0
h_x = -2.0
h_y = 1.0
epsilon = 0.5

##################
# Time parameters
##################

dt_HR = 0.001
N = 10
dt_LR = N*dt_HR
tau_nudge = 0.1
t_end = 50.0
t = np.arange(0.0, t_end, dt_LR)
M = 1
window_length = 100

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

batch_size = window_length - campaign.surrogate.max_lag
batch_size = window_length*K

# set some constants used in online learning
campaign.surrogate.set_online_training_parameters(tau_nudge, dt_LR, window_length)

# a = np.linspace(-5, 10, 100)
# fit_before = campaign.surrogate.predict(a.reshape([1, 100]))

#load training data
data_frame = campaign.load_hdf5_data()
X_train = data_frame['X_data']
Y_train = data_frame['Y_data']
B_train = data_frame['B_data']

# change IC
X_n = X_train[campaign.surrogate.max_lag]
Y_n = Y_train[campaign.surrogate.max_lag]
B_n = B_train[campaign.surrogate.max_lag]

# load reference data
data_frame = campaign.load_hdf5_data(name="load reference data")
X_ref = data_frame['X_data'].flatten()
B_ref = data_frame['B_data'].flatten()

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

# start time integration
idx = 0
for t_i in t:
    
    #store Delta X
    Delta_X = X_n_LR - X_n
    
    #Store LR and HR state at beginning of time step
    X_n_LR_hat = np.copy(X_n_LR)
    X_n_hat = np.copy(X_n)

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

    # solve LR equation
    X_np1_LR, f_n_LR = step_X(X_n_LR, f_nm1_LR, B_n_LR, dt_LR)

    campaign.surrogate.generate_online_training_data(X_n_LR_hat, 
                                                     X_n_LR_hat, X_np1_LR, X_n_hat, X_n)

    # # evaluate the neural network locally
    # B = []
    # for k in range(K): 
    #     B.append(campaign.surrogate.predict(X_n_LR_hat[k]))
    # B = np.array(B).flatten()
 
    # print(len(campaign.surrogate.feat_eng.feat_history[0]))

    B = campaign.surrogate.predict(X_n_LR_hat.reshape([1, K]))

    # update the LR state
    X_np1_LR += B * dt_LR

    # update the neural network every M time steps
    if np.mod(idx, M) == 0 and idx != 0 and idx > campaign.surrogate.window_length:
        campaign.surrogate.train_online(batch_size=batch_size)

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

#############
# Plot PDEs #
#############

post_proc = es.analysis.BaseAnalysis()

fig = plt.figure()
ax = fig.add_subplot(111, xlabel=r'$X_k$')
X_dom_ref, X_pdf_ref = post_proc.get_pdf(X_ref[0:-1:10])
X_dom, X_pdf = post_proc.get_pdf(X_data.flatten()[0:-1:10])
X_dom_LR, X_pdf_LR = post_proc.get_pdf(X_data_LR.flatten()[0:-1:10])
ax.plot(X_dom_ref, X_pdf_ref, 'ko', label='ref')
ax.plot(X_dom, X_pdf, '--g', label='HR')
ax.plot(X_dom_LR, X_pdf_LR, 'b', label='LR')
plt.yticks([])
plt.legend(loc=0)
plt.tight_layout()


#############

colors = ['#636363', '#bdbdbd', '#f0f0f0']
fig = plt.figure(figsize=[5,5])
plt.style.use('seaborn')
ax = fig.add_subplot(111, xlabel='X', ylabel='B')

ax.scatter(X_ref[::200], B_ref[::200], s=8, alpha=1.0, color=colors[0], 
            label=r'correct training data')

ax.scatter(X_train[::200], B_train[::200], s=8, alpha=1.0, color=colors[1],
            label=r'wrong training data')

ax.scatter(X_data[::20], B_data[::20], marker='o', s=8, alpha=0.5, color=colors[2],
            label=r'predicted data')

leg = plt.legend(loc=0)
leg.set_draggable(True)

plt.tight_layout()

plt.show()