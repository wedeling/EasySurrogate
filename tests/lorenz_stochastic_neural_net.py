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
    
    _, bin_idx = surrogate.get_softmax(feat.reshape([1, n_feat]))
    z_ip1 = sampler.draw(bin_idx[0])
    
    return x_ip1, y_ip1, z_ip1, x_dot, y_dot, z_dot

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

def get_binned_Xy(feat, y, n_bins, n_lags):
    
    N_feat = feat.shape[1]
    N = y.size
    
    sub = 1
    X = np.zeros([N - n_lags, N_feat, n_lags])
    for i in range(n_lags):
        begin = i
        end = N - n_lags + i
        for j in range(N_feat):
            X[:, j, i] = feat[begin:end:sub, j]
        
    X = X.reshape([y.size-n_lags, N_feat*n_lags])
    
    bin_idx = np.zeros([y.size, n_bins])
    
    bins = np.linspace(np.min(y), np.max(y), n_bins+1)
    count, _, binnumbers = stats.binned_statistic(y, np.zeros(y.size), statistic='count', bins=bins)
    
    unique_binnumbers = np.unique(binnumbers) 
    
    for i in unique_binnumbers:
        idx = np.where(binnumbers == i)[0]
        bin_idx[idx, i-1] = 1.0    
    
    return X, y[n_lags:], bin_idx[n_lags:], bins

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import easysurrogate as es
from sklearn.neighbors.kde import KernelDensity
from scipy import stats

plt.close('all')

n_steps = 10000
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
    
feat = np.array([X, Y, Z]).T

n_bins = 20
n_lags = 1
feat, y, bin_idx, bins = get_binned_Xy(feat, Z, n_bins, n_lags)

mean_feat = np.mean(feat, axis=0)
std_feat = np.std(feat, axis=0)

n_feat = feat.shape[1]

n_softmax = 1

surrogate = es.methods.ANN(X = feat, y = bin_idx, alpha = 0.001, decay_rate = 0.9, decay_step=10**4, n_out = n_bins*n_softmax, loss = 'cross_entropy', \
                           lamb = 0.01, n_layers = 2, n_neurons=128, activation = 'hard_tanh', activation_out = 'linear', n_softmax = n_softmax, \
                           standardize_y = False, batch_size=512, save=True, aux_vars={'y':y, 'bins':bins})

surrogate.get_n_weights()

surrogate.train(20000, store_loss=True)

if len(surrogate.loss_vals) > 0:
    fig_loss = plt.figure()
    plt.yscale('log')
    plt.plot(surrogate.loss_vals)

########################################
#compute the number of misclassification
########################################

surrogate.compute_misclass_softmax()

#############
#plot results
#############

predicted_bin = np.zeros(surrogate.n_train)

for i in range(surrogate.n_train):
    o_i, idx_max = surrogate.get_softmax(surrogate.X[i].reshape([1, surrogate.n_in]))
    predicted_bin[i] = idx_max

fig = plt.figure(figsize=[10,5])
ax1 = fig.add_subplot(121, title=r'data classification', xlabel=r'$X_i$', ylabel=r'$X_j$')
ax2 = fig.add_subplot(122, title=r'neural net prediction', xlabel=r'$X_i$', ylabel=r'$X_j$')

for j in range(n_bins):
    idx_pred = np.where(predicted_bin == j)[0]
    idx_data = np.where(bin_idx[:, j] == 1.0)[0]
    ax1.plot(surrogate.X[idx_data, 0], surrogate.X[idx_data, 1], 'o', label=r'$\mathrm{bin}\;'+str(j+1)+'$')
    ax2.plot(surrogate.X[idx_pred, 0], surrogate.X[idx_pred, 1], 'o')

ax1.legend()
plt.tight_layout()

sampler = es.methods.SimpleBin(y, bins)

Z_surr = np.zeros(n_steps)

for n in range(n_steps-n_lags):
    _, bin_idx = surrogate.get_softmax(surrogate.X[n].reshape([1, n_feat]))
    Z_surr[n] = sampler.draw(bin_idx[0])


#plot 1-way coupled surrogate result
plt.figure()
plt.plot(Z_surr, '--')
plt.plot(Z)

X_surr = np.zeros(n_steps); Y_surr = np.zeros(n_steps); Z_surr = np.zeros(n_steps) 
X_surr_dot = np.zeros(n_steps); Y_surr_dot = np.zeros(n_steps); Z_surr_dot = np.zeros(n_steps) 

#initial condition
x_i = 0.0; y_i = 1.0; z_i = 1.05

for n in range(n_steps):
    x_i, y_i, z_i, x_dot, y_dot, z_dot = predict_with_surrogate(x_i, y_i, z_i)
    X_surr[n] = x_i
    Y_surr[n] = y_i
    Z_surr[n] = z_i
    X_surr_dot[n] = x_dot
    Y_surr_dot[n] = y_dot
    Z_surr_dot[n] = z_dot

plot_lorenz(X_surr, Y_surr, Z_surr)

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