def predict(x_i, y_i, z_i):
    # Derivatives of the X, Y, Z state
    x_dot, y_dot, z_dot = lorenz(x_i, y_i, z_i)
    x_ip1 = x_i + (x_dot * dt)
    y_ip1 = y_i + (y_dot * dt)
    z_ip1 = z_i + (z_dot * dt)
    
    return x_ip1, y_ip1, z_ip1, x_dot, y_dot, z_dot

def predict_with_surrogate(x_i, y_i, z_i):
    # Derivatives of the X, Y, Z state
    x_dot, y_dot, z_dot = lorenz(x_i, y_i, z_i)
    x_ip1 = x_i + (x_dot * dt)
    y_ip1 = y_i + (y_dot * dt)
    
    feat = np.array([x_i, y_i, z_i])
    feat = (feat - mean_feat)/std_feat
    
    z_ip1 = surrogate.feed_forward(feat.reshape([1, n_feat]))[0][0]
    z_ip1 = z_ip1*std_Z + mean_Z
    
    return x_ip1, y_ip1, z_ip1, x_dot, y_dot, z_dot

def lorenz(x, y, z, s=10, r=28, b=2.667):
    """
    Lorenz 1963 Deterministic Nonperiodic Flow
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def plot_lorenz(xs, ys, zs, title='Lorenz63'):

    fig = plt.figure(title)
    ax = fig.gca(projection='3d')
    
    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    
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

n_steps = 4000
dt = 0.01
X = np.zeros(n_steps); Y = np.zeros(n_steps); Z = np.zeros(n_steps) 
X_dot = np.zeros(n_steps); Y_dot = np.zeros(n_steps); Z_dot = np.zeros(n_steps) 

#initial condition
x_i = 0.0; y_i = 1.0; z_i = 1.05

for n in range(n_steps):
    x_i, y_i, z_i, x_dot, y_dot, z_dot = predict(x_i, y_i, z_i)
    X[n] = x_i
    Y[n] = y_i
    Z[n] = z_i
    X_dot[n] = x_dot
    Y_dot[n] = y_dot
    Z_dot[n] = z_dot
    
plot_lorenz(X, Y, Z)
    
n_lags = 1
feat = np.array([X[0:-n_lags], Y[0:-n_lags], Z[0:-n_lags]]).T
mean_feat = np.mean(feat, axis=0)
std_feat = np.std(feat, axis=0)
mean_Z = np.mean(Z, axis=0)
std_Z = np.std(Z, axis=0)

n_feat = feat.shape[1]

surrogate = es.methods.ANN(X=feat, y=Z[n_lags:], n_layers=3, n_neurons=64, 
                           activation='hard_tanh', batch_size=128,
                           lamb=0.01, decay_step=10**5, decay_rate=0.9)

surrogate.get_n_weights()

surrogate.train(20000, store_loss=True)

if len(surrogate.loss_vals) > 0:
    fig_loss = plt.figure('Loss_function')
    plt.yscale('log')
    plt.plot(surrogate.loss_vals)

Z_surr = np.zeros(n_steps)

for n in range(n_steps-n_lags):
    Z_surr[n] = surrogate.feed_forward(surrogate.X[n].reshape([1,n_feat]))

#plot 1-way coupled surrogate result
plt.figure('one_way_coupled')
plt.plot(Z_surr, '--')
plt.plot((Z - np.mean(Z))/np.std(Z))

X_surr = np.zeros(n_steps); Y_surr = np.zeros(n_steps); Z_surr = np.zeros(n_steps) 
X_surr_dot = np.zeros(n_steps); Y_surr_dot = np.zeros(n_steps); Z_surr_dot = np.zeros(n_steps) 

#initial condition
x_i = 0.0; y_i = 1.0; z_i = 1.05

for n in range(n_steps-n_lags):
    x_i, y_i, z_i, x_dot, y_dot, z_dot = predict_with_surrogate(x_i, y_i, z_i)
    X_surr[n] = x_i
    Y_surr[n] = y_i
    Z_surr[n] = z_i
    X_surr_dot[n] = x_dot
    Y_surr_dot[n] = y_dot
    Z_surr_dot[n] = z_dot

plot_lorenz(X_surr, Y_surr, Z_surr, title='Lorenz63_ANN')

X_dom_surr, X_pde_surr = get_pde(X_surr[0:-1:10])
X_dom, X_pde = get_pde(X[0:-1:10])

Y_dom_surr, Y_pde_surr = get_pde(Y_surr[0:-1:10])
Y_dom, Y_pde = get_pde(Y[0:-1:10])
    
fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(X_dom_surr, X_pde_surr)
ax.plot(X_dom, X_pde, '--k')
ax = fig.add_subplot(122)
ax.plot(Y_dom_surr, Y_pde_surr)
ax.plot(Y_dom, Y_pde, '--k')

plt.show()