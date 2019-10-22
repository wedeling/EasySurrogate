def init_movie():
    for line in lines:
        line.set_data([],[])
    return lines

def animate(s):

    print('Processing frame', s, 'of', sol.shape[0])
    #add the periodic BC to current sample to create a closed plot
    xlist = [theta, theta]
    ylist = [np.append(sol[s, 0:K], sol[s, 0]), np.append(X_data[s, 0:K], X_data[s, 0])]

    #for index in range(0,1):
    for lnum,line in enumerate(lines):
        line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately. 

    return lines

def lorenz96(X, t):
    
    """
    Lorenz96 one-layer model + neural net prediction for B
    """
    
    rhs_X = np.zeros(K)
    
    #first treat boundary cases (k=1, k=2 and k=K)
    rhs_X[0] = -X[K-2]*X[K-1] + X[K-1]*X[1] - X[0] + F
    rhs_X[1] = -X[K-1]*X[0] + X[0]*X[2] - X[1] + F
    rhs_X[K-1] = -X[K-3]*X[K-2] + X[K-2]*X[0] - X[K-1] + F
    
    #treat interior points
    for k in range(2, K-1):
        rhs_X[k] = -X[k-2]*X[k-1] + X[k-1]*X[k+1] - X[k] + F
        
    feat = X
#    feat = (feat - mean_feat)/std_feat
#        
    B = surrogate.feed_forward(feat.reshape([1, K]))[0]
#    B = B*std_y + mean_y
#   
    rhs_X += B
        
    return rhs_X

###################################
# FEATURE ENGINEERING SUBROUTINES #
###################################

def lag_training_data(X, y, lags):
    """    
    Feature engineering: create time-lagged supervised training data X, y
    
    Parameters:
        X: features. Either an array of dimension (n_samples, n_features)
           or a list of arrays of dimension (n_samples, n_features)
           
        y: training target. Array of dimension (n_samples, n_outputs)
        
        lags: list of lists, containing the integer values of lags
              Example: if X=[X_1, X_2] and lags = [[1], [1, 2]], the first 
              feature array X_1 is lagged by 1 (time) step and the second
              by 1 and 2 (time) steps.
              
    Returns:
        X_train, y_trains (arrays), of lagged features and target data. Every
        row of X_train is one (time) lagged feature vector. Every row of y_train
        is a target vector at the next (time) step
    """
    
    #compute the max number of lags in lags
    lags_flattened = list(chain(*lags))
    max_lag = np.max(lags_flattened)
    
    #total number of data samples
    n_samples = y.shape[0]
    
    #if X is one array, add it to a list anyway
    if type(X) == np.ndarray:
        tmp = []
        tmp.append(X)
        X = tmp
    
    #compute target data at next (time) step
    if y.ndim == 2:
        y_train = y[max_lag:, :]
    elif y.ndim == 1:
        y_train = y[max_lag:]
    else:
        print("Error: y must be of dimension (n_samples, ) or (n_samples, n_outputs)")
        return
    
    #a lag list must be specified for every feature in X
    if len(lags) != len(X):
        print('Error: no specified lags for one of the featutes in X')
        return
    
    #compute the lagged features
    C = []
    idx = 0
    for X_i in X:
       
        for lag in lags[idx]:
            begin = max_lag - lag
            end = n_samples - lag
            
            if X_i.ndim == 2:
                C.append(X_i[begin:end, :])
            elif X_i.ndim == 1:
                C.append(X_i[begin:end])
            else:
                print("Error: X must contains features of dimension (n_samples, ) or (n_samples, n_features)")
                return
            idx += 1
            
    #C is a list of lagged features, turn into a single array X_train
    X_train = C[0]
    
    if X_train.ndim == 1:
        X_train = X_train.reshape([y_train.shape[0], 1])    
    
    for X_i in C[1:]:
        
        if X_i.ndim == 1:
            X_i = X_i.reshape([y_train.shape[0], 1])
        
        X_train = np.append(X_train, X_i, axis=1)
            
    return X_train, y_train
    
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
import easysurrogate as es
import h5py, os
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from sklearn.neighbors.kde import KernelDensity
from itertools import chain

plt.close('all')

#Lorenz96 parameters
K = 18
J = 20
F = 10.0
h_x = -1.0
h_y = 1.0
epsilon = 0.5

HOME = os.path.abspath(os.path.dirname(__file__))

#load training data
store_ID = 'L96'
QoI = ['X_data', 'B_data']
h5f = h5py.File(HOME + '/samples/' + store_ID + '.hdf5', 'r')

for q in QoI:
    print(q)
    vars()[q] = h5f[q][:]

#n_lags = 1
#feat = np.append(X_data[0:-n_lags,:], B_data[0:-n_lags, :], axis=1)
#y = B_data[n_lags:, :]

feat = X_data
y = B_data

mean_feat = np.mean(feat, axis=0)
std_feat = np.std(feat, axis=0)
mean_y = np.mean(y, axis=0)
std_y = np.std(y, axis=0)

train = False
if train:
    
    surrogate = es.methods.ANN(X=feat, y=B_data, n_layers=6, n_neurons=32, n_out=K,
                               activation='hard_tanh', batch_size=128,
                               lamb=0.0, decay_step=10**5, decay_rate=0.9, standardize_X=False,
                               standardize_y=False)
    surrogate.train(3000, store_loss=True)
else:
    
    surrogate = es.methods.ANN(X=X_data, y=B_data)
    surrogate.load_ANN()

surrogate.get_n_weights()

#time param
t_end = 10.0
burn = 500
dt = 0.01
t = np.arange(burn*dt, t_end, dt)

X = X_data[0]

#solve system
sol = odeint(lorenz96, X, t)

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
theta = np.linspace(0.0, 2.0*np.pi, K+1)
#create a while in the middle to the plot, scales plot better
ax.set_rorigin(-22)
#remove set radial tickmarks, no labels
ax.set_rgrids([-10, 0, 10], labels=['', '', ''])[0][1]
ax.legend(loc=1)

#if True make a movie of the solution, if not just plot final solution
make_movie = False
if make_movie:

    lines = []
    legends = ['X', 'X_data']
    for i in range(2):
        lobj = ax.plot([],[],lw=2, label=legends[i])[0]
        lines.append(lobj)
    
    anim = FuncAnimation(fig, animate, init_func=init_movie,
                         frames=np.arange(0, sol.shape[0], 10), blit=True)    
    anim.save('demo_ann.gif', writer='imagemagick')
else:
    ax.plot(theta, np.append(sol[-1, 0:K], sol[-1, 0]), label='X')
    
fig = plt.figure()

for k in range(K):
    ax = fig.add_subplot(3, 6, k+1)
    X_dom_surr, X_pde_surr = get_pde(sol[:, k])
    X_dom, X_pde = get_pde(X_data[:, k])
    
    ax.plot(X_dom_surr, X_pde_surr)
    ax.plot(X_dom, X_pde, '--k')
    
plt.show()