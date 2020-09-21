import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from itertools import chain
import matplotlib.pyplot as plt
import easysurrogate as es


def rhs(X_n, s=10, r=28, b=8 / 3):

    x = X_n[0]
    y = X_n[1]
    z = X_n[2]

    f_n = np.zeros(3)

    f_n[0] = s * (y - x)
    f_n[1] = r * x - y - x * z
    f_n[2] = x * y - b * z

    return f_n


def step(X_n, f_nm1):

    # Derivatives of the X, Y, Z state
    f_n = rhs(X_n)

    # Adams Bashforth
    # X_np1 = X_n + dt*(3.0/2.0*f_n - 0.5*f_nm1)

    # Euler
    X_np1 = X_n + dt * f_n

    return X_np1, f_n


def rhs_ccm(X_n, s=10):

    x = X_n

    feat = feat_eng.get_feat_history(max_lag)
    y = ccm.get_sample(feat.reshape([1, N_c]), stochastic=False)
    f_n = s * (y - x)

    return f_n


def step_ccm(X_n, f_nm1):

    # Derivatives of the X, Y, Z state
    f_n = rhs_ccm(X_n)

    # Adams Bashforth
    # X_np1 = X_n + dt*(3.0/2.0*f_n - 0.5*f_nm1)

    # Euler
    X_np1 = X_n + dt * f_n

    feat_eng.append_feat([[X_np1]], max_lag)

    return X_np1, f_n


def plot_lorenz(xs, ys, zs, title='Lorenz63'):

    fig = plt.figure(title)
    ax = fig.gca(projection='3d')

    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)


def one_hot(idx):

    # this should be the non-empty binnumbers
    unique_idx = np.unique(idx)
    B = unique_idx.size
    S = idx.size

    y_idx_binned = np.zeros([S, B])

    count = 0

    for idx_i in unique_idx:
        i = np.where(idx == idx_i)[0]
        y_idx_binned[i, count] = 1.0
        count += 1

    return y_idx_binned


plt.close('all')

###################
# Simulation flags
###################
predict = False  # predict using the learned SGS term
store = False  # store the prediction results

n_steps = 25000
dt = 0.01
X = np.zeros(n_steps)
Y = np.zeros(n_steps)
Z = np.zeros(n_steps)
X_dot = np.zeros(n_steps)
Y_dot = np.zeros(n_steps)
Z_dot = np.zeros(n_steps)

# initial condition
X_n = np.zeros(3)
X_n[0] = 0.0
X_n[1] = 1.0
X_n[2] = 1.05

# initial condition right-hand side
f_nm1 = rhs(X_n)

X = np.zeros([n_steps, 3])
X_dot = np.zeros([n_steps, 3])

for n in range(n_steps):

    #step in time
    X_np1, f_n = step(X_n, f_nm1)

    # update variables
    X_n = X_np1
    f_nm1 = f_n

    X[n, :] = X_n
    X_dot[n, :] = f_n

feat_eng = es.methods.Feature_Engineering()
lags = [[1, 10]]
# n_lags = len(list(chain(*lags)))
max_lag = np.max(list(chain(*lags)))

X_lagged, _ = feat_eng.lag_training_data([X[:, 0]], np.zeros(n_steps), lags)
Y_lagged, _ = feat_eng.lag_training_data([X[:, 1]], np.zeros(n_steps), lags)

ccm = es.methods.CCM(Y_lagged, np.zeros(n_steps), [10, 10], lags)
# N_c = ccm.N_c
# ccm.plot_2D_binning_object()
# ccm.plot_2D_shadow_manifold()
# ccm.compare_convex_hull_volumes()

one_hot_enc = one_hot(ccm.binnumber)

n_out = one_hot_enc.shape[1]

surrogate = es.methods.ANN(X=X_lagged, y=one_hot_enc, n_layers=4, n_neurons=256,
                           n_softmax=1, n_out=n_out, loss='cross_entropy',
                           activation='leaky_relu', batch_size=512,
                           lamb=0.0, decay_step=10**4, decay_rate=0.9,
                           standardize_X=True, standardize_y=False, save=True)

print('===============================')
print('Training Quantized Softmax Network...')

# train network for N_inter mini batches
N_iter = 5000
surrogate.train(N_iter, store_loss=True)

n_train = X_lagged.shape[0]
n_feat = X_lagged.shape[1]

pred = np.zeros(n_train)

for i in range(n_train):
    o_i, idx_max, rvs = surrogate.get_softmax(surrogate.X[i].reshape([1, n_feat]))

    pred[i] = idx_max[0]

plt.figure()

for i in np.unique(pred):

    idx = np.where(pred == i)[0]

    plt.plot(X_lagged[idx, 0], X_lagged[idx, 1], '+')

"""

# feat_eng.estimate_embedding_dimension(X[0:-1:10, 0], 6)

#################################
# Run full model to generate IC #
#################################

X_n = np.zeros(3)
#initial condition of the training data
# X_n[0] = 0.0; X_n[1]  = 1.0; X_n[2] = 1.05

#new initial condition to break symmetry
X_n[0] = 0.20; X_n[1] = 0.75; X_n[2] = 1.0

#initial condition right-hand side
f_nm1 = rhs(X_n)

for n in range(max_lag):

    #step in time
    X_np1, f_n = step(X_n, f_nm1)

    feat_eng.append_feat([[X_np1[0]]], max_lag)

    #update variables
    X_n = X_np1
    f_nm1 = f_n

########################################
# Run the model with the CCM surrogate #
########################################

#number of time steps
n_pred = n_steps - max_lag

#reduce IC to X only
X_n = X_n[0]
f_nm1 = f_nm1[0]

X_surr = np.zeros([n_pred, 1])

for i in range(n_pred):

    #step in time
    X_np1, f_n = step_ccm(X_n, f_nm1)

    #update variables
    X_n = X_np1
    f_nm1 = f_n

    X_surr[i, :] = X_n

#############
# Plot PDEs #
#############

post_proc = es.methods.Post_Processing()

print('===============================')
print('Postprocessing results')

fig = plt.figure(figsize=[4, 4])

ax = fig.add_subplot(111, xlabel=r'$x$')
X_dom_surr, X_pde_surr = post_proc.get_pde(X_surr[0:-1:10, 0])
X_dom, X_pde = post_proc.get_pde(X[0:-1:10, 0])
ax.plot(X_dom, X_pde, 'ko', label='L63')
ax.plot(X_dom_surr, X_pde_surr, label='ANN')
plt.yticks([])
plt.legend(loc=0)

plt.tight_layout()

#############
# Plot ACFs #
#############

fig = plt.figure(figsize=[4, 4])

ax = fig.add_subplot(111, ylabel='ACF X', xlabel='time')
R_data = post_proc.auto_correlation_function(X[:, 0], max_lag = 500)
R_sol = post_proc.auto_correlation_function(X_surr[:, 0], max_lag = 500)
dom_acf = np.arange(R_data.size)*dt
ax.plot(dom_acf, R_data, 'ko', label='L63')
ax.plot(dom_acf, R_sol, label='ANN')
leg = plt.legend(loc=0)

plt.tight_layout()
"""
plt.show()
