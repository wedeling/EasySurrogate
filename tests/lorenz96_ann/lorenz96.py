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


def rhs_X(X, B):
    """
    Compute the right-hand side of X

    Parameters:
        - X (array, size K): large scale variables
        - B (array, size K): SGS term

    returns:
        - rhs_X (array, size K): right-hand side of X
    """

    rhs_X = np.zeros(K)

    # first treat boundary cases (k=1, k=2 and k=K)
    rhs_X[0] = -X[K - 2] * X[K - 1] + X[K - 1] * X[1] - X[0] + F

    rhs_X[1] = -X[K - 1] * X[0] + X[0] * X[2] - X[1] + F

    rhs_X[K - 1] = -X[K - 3] * X[K - 2] + X[K - 2] * X[0] - X[K - 1] + F

    # treat interior points
    for k in range(2, K - 1):
        rhs_X[k] = -X[k - 2] * X[k - 1] + X[k - 1] * X[k + 1] - X[k] + F

    rhs_X += B

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


def step_X(X_n, f_nm1, B):
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
    f_n = rhs_X(X_n, B)

    # adams bashforth
    X_np1 = X_n + dt * (1.5 * f_n - 0.5 * f_nm1)

    return X_np1, f_n


def step_Y(Y_n, g_nm1, X_n):
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

    return Y_np1, g_n, multistep_rhs

#####################
# Other subroutines #
#####################


def animate(s):
    """
    Generate a movie frame
    """
    plt1 = ax.plot(theta, np.append(X_data[s, :], X_data[s, 0]), 'b', label='X')
    plt2 = ax.plot(theta_Y, np.append(Y_data[s, :].flatten(), Y_data[s, 0, 0]), 'r', label='Y')

    if s == 0:
        ax.legend(loc=1, fontsize=9)

    ims.append((plt1[0], plt2[0],))

###############
# Main program
###############


plt.close('all')

#####################
# Lorenz96 parameters
#####################

K = 18
J = 20
F = 10.0
h_x = -1.0
h_y = 1.0
epsilon = 0.5

##################
# Time parameters
##################
dt = 0.01
t_end = 1000.0
t = np.arange(0.0, t_end, dt)

###################
# Simulation flags
###################
make_movie = False  # make a movie
store = True  # store the prediction results

# equilibrium initial condition for X, zero IC for Y
X_n = np.ones(K) * F
X_n[10] += 0.01  # add small perturbation to 10th variable

# initial condition small-scale variables
Y_n = np.zeros([J, K])
B_n = h_x * np.mean(Y_n, axis=0)

# initial right-hand sides
f_nm1 = rhs_X(X_n, B_n)

g_nm1 = np.zeros([J, K])
for k in range(K):
    g_nm1[:, k] = rhs_Y_k(Y_n, X_n[k], k)

# allocate memory for solutions
X_data = np.zeros([t.size, K])
Y_data = np.zeros([t.size, J, K])
B_data = np.zeros([t.size, K])

# start time integration
idx = 0
for t_i in t:
    # solve small-scale equation
    Y_np1, g_n, multistep_n = step_Y(Y_n, g_nm1, X_n)

    # compute SGS term
    B_n = h_x * np.mean(Y_n, axis=0)

    # solve large-scale equation
    X_np1, f_n = step_X(X_n, f_nm1, B_n)

    # store solutions
    X_data[idx, :] = X_np1
    Y_data[idx, :] = Y_np1
    B_data[idx, :] = B_n
    idx += 1

    # update variables
    X_n = np.copy(X_np1)
    Y_n = np.copy(Y_np1)
    f_nm1 = np.copy(f_n)
    g_nm1 = np.copy(g_n)

    if np.mod(idx, 1000) == 0:
        print('t =', np.around(t_i, 1), 'of', t_end)

# plot results
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
theta = np.linspace(0.0, 2.0 * np.pi, K + 1)
theta_Y = np.linspace(0.0, 2.0 * np.pi, J * K + 1)
# create a while in the middle to the plot, scales plot better
ax.set_rorigin(-22)
# remove set radial tickmarks, no labels
ax.set_rgrids([-10, 0, 10], labels=['', '', ''])[0][1]
ax.legend(loc=1)

burn = 500

post_proc = es.analysis.BaseAnalysis()

# store results
if store:
    # store results
    samples = {}
    QoI = {'X_data', 'Y_data', 'B_data'}

    for q in QoI:
        print('Saving', q)
        samples[q] = eval(q)[burn:, :]

    campaign = es.Campaign()
    campaign.store_data_to_hdf5(samples)

# if True make a movie of the solution, if not just plot final solution
if make_movie:

    print('===============================')
    print('Making movie...')

    ims = []
    n_movie = 500

    for s in range(n_movie):
        animate(s)

    # make a movie of all frame in 'ims'
    im_ani = animation.ArtistAnimation(fig, ims, interval=80,
                                       repeat_delay=2000, blit=True)
    im_ani.save('./l96.gif')

    print('done')
else:
    ax.plot(theta, np.append(X_data[-1, :], X_data[-1, 0]), label='X')
    ax.plot(theta_Y, np.append(Y_data[-1, :].flatten(), Y_data[-1, 0, 0]), label='Y')

#############
# Plot PDEs #
#############

fig = plt.figure()
ax = fig.add_subplot(111, xlabel=r'$X_k$')

X_dom, X_pde = post_proc.get_pdf(X_data.flatten()[0:-1:10])

ax.plot(X_dom, X_pde, 'ko', label='L96')

plt.yticks([])

plt.legend(loc=0)

plt.tight_layout()

plt.show()
