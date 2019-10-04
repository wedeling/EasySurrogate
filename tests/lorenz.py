def predict(x_i, y_i, z_i):
    # Derivatives of the X, Y, Z state
    x_dot, y_dot, z_dot = lorenz(x_i, y_i, z_i)
    x_ip1 = x_i + (x_dot * dt)
    y_ip1 = y_i + (y_dot * dt)
    z_ip1 = z_i + (z_dot * dt)
    
    return x_ip1, y_ip1, z_ip1

def predict_with_surrogate(x_i, y_i, z_i):
    # Derivatives of the X, Y, Z state
    x_dot, y_dot, z_dot = lorenz(x_i, y_i, z_i)
    x_ip1 = x_i + (x_dot * dt)
    y_ip1 = y_i + (y_dot * dt)
    
    c_i = surrogate.get_covar()
    z_ip1 = surrogate.sample(c_i)

    covar_ip1 = np.array([x_ip1, y_ip1, z_ip1])
    surrogate.append_covar(covar_ip1.reshape([1,3])) 
    
    return x_ip1, y_ip1, z_ip1

def lorenz(x, y, z, s=10, r=28, b=2.667):
    """
    Lorenz 1963 Deterministic Nonperiodic Flow
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def plot_lorenz(xs, ys, zs):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    
def init_condvar_data(c, r, lags):
    
    N_c = len(list(chain(*lags)))
    R = r[max_lag:]
    
    C = np.zeros([R.size, N_c])
    idx = 0
    
    for i in range(len(lags)):
        for lag in lags[i]:
            C[:, idx] = lag_array(c[:, i], lag, max_lag)
            idx += 1
            
    return C, R
    
#make part of Resampler
def lag_array(x, n_lag, max_lag):

    begin = max_lag - n_lag
    end = x.size - n_lag
    
    return x[begin:end]

def get_pde(X, Npoints = 100):

#    kernel = stats.gaussian_kde(X, bw_method='scott')
#    x = np.linspace(np.min(X), np.max(X), Npoints)
#    pde = kernel.evaluate(x)
#    return x, pde
    
    X_min = np.min(X)
    X_max = np.max(X)
    bandwidth = (X_max-X_min)/40
    
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X.reshape(-1, 1))
    domain = np.linspace(X_min, X_max, Npoints).reshape(-1, 1)
    log_dens = kde.score_samples(domain)
    
    return domain, np.exp(log_dens)   

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import easysurrogate as es
from itertools import chain
from sklearn.neighbors.kde import KernelDensity

plt.close('all')

n_steps = 100000
dt = 0.01
X = np.zeros(n_steps); Y = np.zeros(n_steps); Z = np.zeros(n_steps) 

#initial condition
x_i = 0.0; y_i = 1.0; z_i = 1.05

for n in range(n_steps):
    x_i, y_i, z_i = predict(x_i, y_i, z_i)
    X[n] = x_i
    Y[n] = y_i
    Z[n] = z_i
    
plot_lorenz(X, Y, Z)
    
lags = [[1], [1], [1]]
max_lag = np.max(list(chain(*lags)))
n_bins = 20
N = 1

c = np.array([X, Y, Z]).T
r = Z
C, R = init_condvar_data(c, r, lags)

surrogate = es.methods.Resampler(C, R, N, n_bins, lags)
surrogate.print_bin_info()
surrogate.plot_2D_binning_object()
#
#initial condition
x_i = 0.0; y_i = 1.0; z_i = 1.05

n_steps = 100000
X_surr = np.zeros(n_steps); Y_surr = np.zeros(n_steps); Z_surr = np.zeros(n_steps) 

#initialize the conditional variables by running the full model max_lag
#times
for n in range(max_lag):
    x_i, y_i, z_i = predict(x_i, y_i, z_i)
    covar_i = np.array([x_i, y_i, z_i])
    surrogate.append_covar(covar_i.reshape([1,3]))
    X_surr[n] = x_i
    Y_surr[n] = y_i
    Z_surr[n] = z_i
    
for n in range(max_lag, n_steps):
    x_i, y_i, z_i = predict_with_surrogate(x_i, y_i, z_i)
    X_surr[n] = x_i
    Y_surr[n] = y_i
    Z_surr[n] = z_i

plot_lorenz(X_surr, Y_surr, Z_surr)

dom_surr, X_pde_surr = get_pde(X_surr[0:-1:10])
dom, X_pde = get_pde(X[0:-1:10])
    
fig = plt.figure()
plt.plot(dom_surr, X_pde_surr)
plt.plot(dom, X_pde, '--k')

plt.show()