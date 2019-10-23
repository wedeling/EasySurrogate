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
        
    feat = feat_eng.get_feat_history()
#    feat = (feat - mean_feat)/std_feat
#
#    feat = X        
    B = surrogate.feed_forward(feat.reshape([1, feat.size])).flatten()
#    B = B*std_y + mean_y
#   
    feat_eng.append_feat([X])
   
    rhs_X += B
        
    return rhs_X


def ACF(X, max_lag_time):
    
    lags = np.arange(1, max_lag)
    R = np.zeros(lags.size)
    
    idx = 0
    
    #for every lag, compute autocorrelation:
    # R = E[(X_t - mu_t)*(X_s - mu_s)]/(std_t*std_s)
    for lag in lags:
    
        X_t = X[0:-lag]
        X_s = X[lag:]
    
        mu_t = np.mean(X_t)
        std_t = np.std(X_t)
        mu_s = np.mean(X_s)
        std_s = np.std(X_s)
    
        R[idx] = np.mean((X_t - mu_t)*(X_s - mu_s))/(std_t*std_s)
        idx += 1

    return R
   
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

feat_eng = es.methods.Feature_Engineering(X_data, B_data)

lags = [[1]]
max_lag = np.max(list(chain(*lags)))
X_train, y_train = feat_eng.lag_training_data([X_data], lags = lags)

train = True
if train:
    
    surrogate = es.methods.ANN(X=X_train, y=y_train, n_layers=3, n_neurons=128, n_out=K,
                               activation='hard_tanh', batch_size=128,
                               lamb=0.01, decay_step=10**5, decay_rate=0.9, standardize_X=False,
                               standardize_y=False)
    surrogate.train(20000, store_loss=True)
else:    
    surrogate = es.methods.ANN(X=X_train, y=y_train)
    surrogate.load_ANN()

surrogate.get_n_weights()

#time param
t_end = 1000.0
burn = 500
dt = 0.01
t = np.arange(burn*dt, t_end, dt)

for i in range(max_lag):
    feat_eng.append_feat([X_data[i]])

X = X_data[max_lag]

#solve system
sol = odeint(lorenz96, X, t)

post_proc = es.methods.Post_Processing()

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
 
#############   
# Plot PDEs #
#############
    
fig = plt.figure()
ax = fig.add_subplot(111)

X_dom_surr, X_pde_surr = post_proc.get_pde(sol.flatten())
X_dom, X_pde = post_proc.get_pde(X_data.flatten()[0:-1:10])

ax.plot(X_dom_surr, X_pde_surr)
ax.plot(X_dom, X_pde, '--k')

plt.tight_layout()

#############   
# Plot ACFs #
#############

fig = plt.figure()
ax = fig.add_subplot(111, ylabel='ACF', xlabel='time')

R_data = post_proc.auto_correlation_function(X_data[:,0], max_lag=1000)
R_sol = post_proc.auto_correlation_function(sol[:, 0], max_lag=1000)

dom = np.arange(R_data.size)*dt

ax.plot(dom, R_data, '--k', label='L96')
ax.plot(dom, R_sol, label='ANN')

leg = plt.legend(loc=0)

plt.tight_layout()

plt.show()
