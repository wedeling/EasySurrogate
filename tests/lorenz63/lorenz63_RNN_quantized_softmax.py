import numpy as np
from itertools import chain
import easysurrogate as es
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def rhs(X_n, s, r=28, b=2.667):
    """
    Lorenz 1963 Deterministic Nonperiodic Flow
    """
    x = X_n[0]
    y = X_n[1]
    z = X_n[2]

    f_n = np.zeros(3)

    f_n[0] = s * (y - x)
    f_n[1] = r * x - y - x * z
    f_n[2] = x * y - b * z

    return f_n


def rhs_surrogate(X_n, y_nm1, r_nm1, s=10):

    feat = feat_eng.get_feat_history().reshape([1, n_feat])
    feat = (feat - mean_feat) / std_feat
    o_i, idx_max, idx_rvs = surrogate.get_softmax(feat.reshape([1, n_feat]))
    y_n = sampler.resample(idx_max)[0]

    # beta = 0.9
    # r_n = beta*r_nm1 + (1.0 - beta)*y_n

    tau1 = 100.0
    # tau2 = 1.0
    r_n = r_nm1 + dt * tau1 * (y_n - r_nm1)

    f_n = s * (y_n - X_n)
    # f_n[1] = r*x - y - x*z
    # z_dot = x*y - b*z

    return f_n, y_n, r_n


def step(X_n, f_nm1):

    # Derivatives of the X, Y, Z state
    f_n = rhs(X_n, sigma)

    # Adams Bashforth
    # X_np1 = X_n + dt*(3.0/2.0*f_n - 0.5*f_nm1)

    # Euler
    X_np1 = X_n + dt * f_n

    return X_np1, f_n


def step_with_surrogate(X_n, y_nm1, r_nm1, f_nm1):

    # Derivatives of the X, Y, Z state
    f_n, y_n, r_n = rhs_surrogate(X_n, y_nm1, r_nm1)

    # Adams Bashforth
    # X_np1 = X_n + dt*(3.0/2.0*f_n - 0.5*f_nm1)

    # Euler
    X_np1 = X_n + dt * f_n

    feat_eng.append_feat([[X_np1], [y_n]], 1)

    return X_np1, y_n, r_n, f_n


def plot_lorenz(ax, xs, ys, zs, title='Lorenz63'):

    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)


plt.close('all')

n_steps = 10**5
dt = 0.01
sigma = 10.0
alpha = 1.0 - sigma * dt

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
f_nm1 = rhs(X_n, sigma)

X = np.zeros([n_steps, 3])
X_dot = np.zeros([n_steps, 3])

for n in range(n_steps):

    #step in time
    X_np1, f_n = step(X_n, f_nm1)

    # makes it fail!
    # X[n, :] = X_n

    # update variables
    X_n = X_np1
    f_nm1 = f_n

    X[n, :] = X_n
    X_dot[n, :] = f_n

feat_eng = es.methods.Feature_Engineering()

# X_train = X[:, 0:1].reshape([n_steps, 1])
# y_train = X[:, 1].reshape([n_steps, 1])

X_train, y_train = feat_eng.lag_training_data([X[:, 0], X[:, 1]], X[:, 1], [[0], [1]])

n_train = X_train.shape[0]
n_feat = X_train.shape[1]

n_bins = 10
feat_eng.bin_data(y_train, n_bins)
sampler = es.methods.SimpleBin(feat_eng)


surrogate = es.methods.RNN(X_train, feat_eng.y_idx_binned, alpha=0.001,
                           decay_rate=0.9, decay_step=10**4, activation='tanh',
                           bias=True, n_neurons=100, n_layers=2, sequence_size=100,
                           n_softmax=1, n_out=n_bins,
                           training_mode='offline', loss='cross_entropy',
                           save=False, param_specific_learn_rate=True, standardize_y=False)

surrogate.train(5000)

mean_feat = surrogate.X_mean
std_feat = surrogate.X_std

test = []
surrogate.window_idx = 0
S = X_train.shape[0]

X_test = (X_train - surrogate.X_mean) / surrogate.X_std
# surrogate.clear_history()

I = 10000

for i in range(I):
    o_i, idx_max, idx_rvs = surrogate.get_softmax(X_test[i].reshape([1, n_feat]))
    test.append(sampler.resample(idx_max))
    # test.append(surrogate.feed_forward())
# test = list(chain(*test))

plt.plot(test)
plt.plot(y_train[0:I], 'ro')

X_surr = np.zeros([n_steps, 1])
X_surr_dot = np.zeros([n_steps, 1])

# initial condition, pick a random point from the data
idx_start = 1
X_n = X[idx_start, 0]
f_nm1 = X_dot[idx_start - 1, 0]
y_nm1 = X[idx_start - 1, 1]
r_nm1 = X[idx_start - 1, 1]

# features are time lagged, use the data to create initial feature set
for i in range(1):
    j = idx_start - 1 + i
    feat_eng.append_feat([[X[j, 0]], [X[j, 1]]], 1)

outputs = []
outputs_smooth = []

for n in range(n_train):

    X_surr[n, :] = X_n

    #step in time
    X_np1, y_n, r_n, f_n = step_with_surrogate(X_n, y_nm1, r_nm1, f_nm1)

    outputs.append(y_n)
    outputs_smooth.append(r_n)

    X_surr_dot[n, :] = f_n

    # update variables
    X_n = X_np1
    f_nm1 = f_n
    r_nm1 = r_n

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
R_data = post_proc.auto_correlation_function(X[:, 0], max_lag=500)
R_sol = post_proc.auto_correlation_function(X_surr[:, 0], max_lag=500)
dom_acf = np.arange(R_data.size) * dt
ax.plot(dom_acf, R_data, 'ko', label='L63')
ax.plot(dom_acf, R_sol, label='ANN')
leg = plt.legend(loc=0)

plt.tight_layout()

plt.show()
