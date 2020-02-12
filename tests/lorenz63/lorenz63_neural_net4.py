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

def rhs_surrogate(X_n, s=10):

    feat = feat_eng.get_feat_history().flatten()
    feat = (feat - mean_feat)/std_feat
    y = surrogate.feed_forward(feat.reshape([1, n_feat]))[0]
    y = y*std_data + mean_data

    f_n = s*(y - X_n)
    # f_n[1] = r*x - y - x*z
    # z_dot = x*y - b*z
    
    return f_n

def step(X_n, f_nm1):
    
    # Derivatives of the X, Y, Z state
    f_n = rhs(X_n)

    # Adams Bashforth
    # X_np1 = X_n + dt*(3.0/2.0*f_n - 0.5*f_nm1)
    
    # Euler
    X_np1 = X_n + dt*f_n
    
    return X_np1, f_n

def step_with_surrogate(X_n, f_nm1):

    # Derivatives of the X, Y, Z state
    f_n = rhs_surrogate(X_n)

    # Adams Bashforth
    # X_np1 = X_n + dt*(3.0/2.0*f_n - 0.5*f_nm1)
    
    # Euler
    X_np1 = X_n + dt*f_n
   
    feat_eng.append_feat([[X_np1[0]]], max_lag)
    
    return X_np1, f_n

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

    #makes it fail!
    # X[n, :] = X_n

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
lags = [range(1, 100, 5)]
max_lag = np.max(list(chain(*lags)))

X_train, y_train = feat_eng.lag_training_data([X[:, 0]], X[:, 1], lags = lags)
# mean_feat, std_feat = feat_eng.moments_lagged_features([X, Y, Z], lags)
mean_feat = np.mean(X_train, axis = 0)
std_feat = np.std(X_train, axis = 0)
mean_data = np.mean(y_train, axis = 0)
std_data = np.std(y_train, axis = 0)
    
n_feat = X_train.shape[1]
n_train = X_train.shape[0]

surrogate = es.methods.ANN(X=X_train, y=y_train, n_layers=3, n_neurons=128, n_out=1, 
                           activation='hard_tanh', batch_size=128,
                           lamb=0.0, decay_step=10**4, decay_rate=0.9, save = False)
surrogate.get_n_weights()

surrogate.train(20000, store_loss=True)

X_surr = np.zeros([n_steps, 1])
X_surr_dot = np.zeros([n_steps, 1])

#initial condition, pick a random point from the data
idx_start = np.random.randint(max_lag, n_train)
X_n = X[idx_start, 0]
f_nm1 = X_dot[idx_start - 1, 0]

#features are time lagged, use the data to create initial feature set
for i in range(max_lag):
    j = idx_start - max_lag + 1
    feat_eng.append_feat([[X[j, 0]]], max_lag)

for n in range(n_train):
    
    X_surr[n, :] = X_n
        
    #step in time
    X_np1, f_n = step_with_surrogate(X_n, f_nm1)

    X_surr_dot[n, :] = f_n

    #update variables
    X_n = X_np1
    f_nm1 = f_n  
    
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

plt.show()