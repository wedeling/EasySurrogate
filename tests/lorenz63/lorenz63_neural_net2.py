import numpy as np
from itertools import chain
import easysurrogate as es
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def predict(x_i, y_i, z_i):
    # Derivatives of the X, Y, Z state
    x_dot, y_dot, z_dot = lorenz(x_i, y_i, z_i)
    x_ip1 = x_i + (x_dot * dt)
    y_ip1 = y_i + (y_dot * dt)
    z_ip1 = z_i + (z_dot * dt)

    return x_ip1, y_ip1, z_ip1, x_dot, y_dot, z_dot


def predict_with_surrogate(x_i, y_i, z_i):
    # Derivatives of the X, Y, Z state
    x_dot, y_dot, z_dot = lorenz_surrogate(x_i, y_i, z_i)
    x_ip1 = x_i + (x_dot * dt)
    y_ip1 = y_i + (y_dot * dt)
    z_ip1 = z_i + (z_dot * dt)

    feat_eng.append_feat([[x_ip1], [y_ip1]], max_lag)

    return x_ip1, y_ip1, z_ip1, x_dot, y_dot, z_dot


def lorenz(x, y, z, s=10, r=28, b=2.667):
    """
    Lorenz 1963 Deterministic Nonperiodic Flow
    """
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


def lorenz_surrogate(x, y, z, s=10, r=28):
    """
    Lorenz 1963 Deterministic Nonperiodic Flow
    """
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    # z_dot = x*y - b*z

    feat = feat_eng.get_feat_history().flatten()
    feat = (feat - mean_feat) / std_feat
    z_dot = surrogate.feed_forward(feat.reshape([1, n_feat]))[0]
    z_dot = z_dot * std_data + mean_data

    return x_dot, y_dot, z_dot


def plot_lorenz(ax, xs, ys, zs, title='Lorenz63'):

    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)


plt.close('all')

n_steps = 10**5
dt = 0.01
X = np.zeros(n_steps)
Y = np.zeros(n_steps)
Z = np.zeros(n_steps)
X_dot = np.zeros(n_steps)
Y_dot = np.zeros(n_steps)
Z_dot = np.zeros(n_steps)

# initial condition
x_i = 0.0
y_i = 1.0
z_i = 1.05

for n in range(n_steps):
    x_i, y_i, z_i, x_dot, y_dot, z_dot = predict(x_i, y_i, z_i)
    X[n] = x_i
    Y[n] = y_i
    Z[n] = z_i
    X_dot[n] = x_dot
    Y_dot[n] = y_dot
    Z_dot[n] = z_dot

#####################
# Network parameters
#####################

# Feature engineering object - loads data file
feat_eng = es.methods.Feature_Engineering()

# Lag features as defined in 'lags'
lags = [range(1, 22, 10), range(1, 22, 10)]
max_lag = np.max(list(chain(*lags)))

X_train, y_train = feat_eng.lag_training_data([X, Y], Z_dot, lags=lags)
# mean_feat, std_feat = feat_eng.moments_lagged_features([X, Y, Z], lags)
mean_feat = np.mean(X_train, axis=0)
std_feat = np.std(X_train, axis=0)
mean_data = np.mean(y_train, axis=0)
std_data = np.std(y_train, axis=0)

n_feat = X_train.shape[1]
n_train = X_train.shape[0]

surrogate = es.methods.ANN(X=X_train, y=y_train, n_layers=3, n_neurons=64, n_out=1,
                           activation='hard_tanh', batch_size=128,
                           lamb=0.0, decay_step=10**4, decay_rate=0.9, save=False)
surrogate.get_n_weights()

surrogate.train(20000, store_loss=True)

X_surr = np.zeros(n_steps)
Y_surr = np.zeros(n_steps)
Z_surr = np.zeros(n_steps)
X_surr_dot = np.zeros(n_steps)
Y_surr_dot = np.zeros(n_steps)
Z_surr_dot = np.zeros(n_steps)

# initial condition, pick a random point from the data
idx_start = np.random.randint(max_lag, n_train)
x_i = X[idx_start]
y_i = Y[idx_start]
z_i = Z[idx_start]

# features are time lagged, use the data to create initial feature set
for i in range(max_lag):
    j = idx_start - max_lag + 1
    feat_eng.append_feat([[X[j]], [Y[j]]], max_lag)

for n in range(n_train):
    x_i, y_i, z_i, x_dot, y_dot, z_dot = predict_with_surrogate(x_i, y_i, z_i)
    X_surr[n] = x_i
    Y_surr[n] = y_i
    Z_surr[n] = z_i
    X_surr_dot[n] = x_dot
    Y_surr_dot[n] = y_dot
    Z_surr_dot[n] = z_dot

###################
# Plot attractors #
###################

fig = plt.figure(figsize=[8, 4])
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

plot_lorenz(ax1, X, Y, Z, title='Lorenz63 data')
plot_lorenz(ax2, X_surr, Y_surr, Z_surr, title='Lorenz63 neural network')

plt.tight_layout()

#############
# Plot PDEs #
#############

print('===============================')
print('Postprocessing results')

fig = plt.figure()
ax = fig.add_subplot(111, xlabel=r'$X_k$')

post_proc = es.methods.Post_Processing()
X_dom_surr, X_pde_surr = post_proc.get_pde(X_surr.flatten()[0:-1:10])
X_dom, X_pde = post_proc.get_pde(X.flatten()[0:-1:10])

ax.plot(X_dom, X_pde, 'ko', label='L96')
ax.plot(X_dom_surr, X_pde_surr, label='ANN')

plt.yticks([])

plt.legend(loc=0)

plt.tight_layout()

#############
# Plot ACFs #
#############

fig = plt.figure(figsize=[12, 4])

ax = fig.add_subplot(131, ylabel='ACF X', xlabel='time')
R_data = post_proc.auto_correlation_function(X, max_lag=500)
R_sol = post_proc.auto_correlation_function(X_surr, max_lag=500)
dom_acf = np.arange(R_data.size) * dt
ax.plot(dom_acf, R_data, 'ko', label='L63')
ax.plot(dom_acf, R_sol, label='ANN')
leg = plt.legend(loc=0)

ax = fig.add_subplot(132, ylabel='ACF Y', xlabel='time')
R_data = post_proc.auto_correlation_function(Y, max_lag=500)
R_sol = post_proc.auto_correlation_function(Y_surr, max_lag=500)
dom_acf = np.arange(R_data.size) * dt
ax.plot(dom_acf, R_data, 'ko', label='L63')
ax.plot(dom_acf, R_sol, label='ANN')
leg = plt.legend(loc=0)

ax = fig.add_subplot(133, ylabel='ACF Z', xlabel='time')
R_data = post_proc.auto_correlation_function(Z, max_lag=500)
R_sol = post_proc.auto_correlation_function(Z_surr, max_lag=500)
dom_acf = np.arange(R_data.size) * dt
ax.plot(dom_acf, R_data, 'ko', label='L63')
ax.plot(dom_acf, R_sol, label='ANN')
leg = plt.legend(loc=0)

plt.tight_layout()

plt.show()
