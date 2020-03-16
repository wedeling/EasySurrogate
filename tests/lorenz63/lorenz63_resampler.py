def rhs(X_n, s=10, r=28, b=2.667):
    """
    Lorenz 1963 Deterministic Nonperiodic Flow
    """
    x = X_n[0]; y = X_n[1]; z = X_n[2]
    
    f_n = np.zeros(3)
    
    f_n[0] = s*(y - x)
    f_n[1] = r*x - y - x*z
    f_n[2] = x*y - b*z
    
    return f_n

def rhs_surrogate(X_n, y_nm1, s=10):

    feat = feat_eng.get_feat_history(max_lag).reshape([1, n_feat])
    y_n = surrogate.get_sample(feat)[0][0]
    y_n_mean = surrogate.get_mean_sample(feat)[0][0]
    
    # beta = 0.9
    # r_n = beta*r_nm1 + (1.0 - beta)*y_n
    
    # tau1 = 100.0
    # # tau2 = 1.0
    # r_n = r_nm1 + dt*tau1*(y_n_mean - r_nm1) 
    
    f_n = s*(y_n - X_n)
    # f_n[1] = r*x - y - x*z
    # z_dot = x*y - b*z
    
    return f_n, y_n

def step(X_n, f_nm1):
    
    # Derivatives of the X, Y, Z state
    f_n = rhs(X_n)

    # Adams Bashforth
    # X_np1 = X_n + dt*(3.0/2.0*f_n - 0.5*f_nm1)
    
    # Euler
    X_np1 = X_n + dt*f_n
    
    return X_np1, f_n

def step_with_surrogate(X_n, y_nm1, f_nm1):

    # Derivatives of the X, Y, Z state
    f_n, y_n = rhs_surrogate(X_n, y_nm1)

    # Adams Bashforth
    # X_np1 = X_n + dt*(3.0/2.0*f_n - 0.5*f_nm1)
    
    # Euler
    X_np1 = X_n + dt*f_n
   
    feat_eng.append_feat([[X_np1], [y_n]], max_lag)
    
    return X_np1, y_n, f_n

def plot_lorenz(ax, xs, ys, zs, title='Lorenz63'):
    
    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)
    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import easysurrogate as es
from itertools import chain

plt.close('all')

n_steps = 10**5
dt = 0.01
X = np.zeros(n_steps); Y = np.zeros(n_steps); Z = np.zeros(n_steps) 
X_dot = np.zeros(n_steps); Y_dot = np.zeros(n_steps); Z_dot = np.zeros(n_steps) 

#initial condition
X_n = np.zeros(3)
X_n[0] = 0.0; X_n[1]  = 1.0; X_n[2] = 1.05

#initial condition right-hand side
f_nm1 = rhs(X_n)

X = np.zeros([n_steps, 3])
X_dot = np.zeros([n_steps, 3])

for n in range(n_steps):
    
    #step in time
    X_np1, f_n = step(X_n, f_nm1)

    #update variables
    X_n = X_np1
    f_nm1 = f_n

    X[n, :] = X_n
    X_dot[n, :] = f_n
    
#####################
# Network parameters
#####################

#Feature engineering object - loads data file
feat_eng = es.methods.Feature_Engineering()

#Lag features as defined in 'lags'
lags = [[1, 10], [1]]
max_lag = np.max(list(chain(*lags)))
X_train, y_train = feat_eng.lag_training_data([X[:, 0], X[:, 1]], X[:, 1], lags = lags)

#number of input features
n_train = X_train.shape[0]
n_feat = X_train.shape[1]

#number of bins
n_bins = [10, 10, 10]

surrogate = es.methods.Resampler(X_train, y_train.flatten(), 1, n_bins, lags)
surrogate.print_bin_info()
surrogate.plot_2D_binning_object()

n_predict = n_train
X_surr = np.zeros([n_predict, 1])
X_surr_dot = np.zeros([n_predict, 1])

#initial condition, pick a random point from the data
idx_start = np.random.randint(max_lag, n_train)
idx_start = max_lag
X_n = X[idx_start, 0]
f_nm1 = X_dot[idx_start - 1, 0]
y_nm1 = X[idx_start - 1, 1]
r_nm1 = X[idx_start - 1, 1]

#features are time lagged, use the data to create initial feature set
for i in range(max_lag):
    j = idx_start - max_lag + i
    feat_eng.append_feat([[X[j, 0]], [X[j, 1]]], max_lag)

outputs = []
outputs_smooth = []

for n in range(n_predict):
    
    X_surr[n, :] = X_n
        
    #step in time
    X_np1, y_n, f_n = step_with_surrogate(X_n, y_nm1, f_nm1)

    outputs.append(y_n)

    X_surr_dot[n, :] = f_n

    #update variables
    X_n = X_np1
    f_nm1 = f_n  
    
#############   
# Plot PDEs #
#############

outputs = np.array(outputs)
post_proc = es.methods.Post_Processing()

print('===============================')
print('Postprocessing results')

fig = plt.figure(figsize=[8, 4])

ax = fig.add_subplot(121, xlabel=r'$x$')
X_dom_surr, X_pde_surr = post_proc.get_pde(X_surr[0:-1:10, 0])
X_dom, X_pde = post_proc.get_pde(X[0:-1:10, 0])
ax.plot(X_dom, X_pde, 'ko', label='L63')
ax.plot(X_dom_surr, X_pde_surr, label='Resampler')
plt.yticks([])
plt.legend(loc=0)

ax = fig.add_subplot(122, xlabel=r'$y$')
X_dom_surr, X_pde_surr = post_proc.get_pde(outputs[0:-1:10])
X_dom, X_pde = post_proc.get_pde(X[0:-1:10, 1])
ax.plot(X_dom, X_pde, 'ko', label='L63')
ax.plot(X_dom_surr, X_pde_surr, label='Resampler')
plt.yticks([])
plt.legend(loc=0)

plt.tight_layout()

#############   
# Plot ACFs #
#############

fig = plt.figure(figsize=[8, 4])

ax = fig.add_subplot(121, ylabel='ACF X', xlabel='time')
R_data = post_proc.auto_correlation_function(X[:, 0], max_lag = 500)
R_sol = post_proc.auto_correlation_function(X_surr[:, 0], max_lag = 500)
dom_acf = np.arange(R_data.size)*dt
ax.plot(dom_acf, R_data, 'ko', label='L63')
ax.plot(dom_acf, R_sol, label='Resampler')
leg = plt.legend(loc=0)

ax = fig.add_subplot(122, ylabel='ACF Y', xlabel='time')
R_data = post_proc.auto_correlation_function(X[:, 1], max_lag = 500)
R_sol = post_proc.auto_correlation_function(outputs, max_lag = 500)
dom_acf = np.arange(R_data.size)*dt
ax.plot(dom_acf, R_data, 'ko', label='L63')
ax.plot(dom_acf, R_sol, label='Resampler')
leg = plt.legend(loc=0)

plt.tight_layout()

plt.show()