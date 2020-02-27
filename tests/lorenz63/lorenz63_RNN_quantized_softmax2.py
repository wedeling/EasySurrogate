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

def rhs_surrogate(X_n, s=10, r=28):

    x = X_n[0]; y = X_n[1]#; z = X_n[2]
    
    feat = np.array([X_n]).flatten()
    feat = (feat - mean_feat)/std_feat
    o_i, idx_max, idx_rvs = surrogate.get_softmax(feat.reshape([1, n_feat]))
    xz = sampler.resample_mean(idx_max)
    
    # y = Y_n + y_dot*dt
    f_n = np.zeros(2)

    f_n[0] = s*(y - x)
    f_n[1] = r*x - y - xz
    # z_dot = x*y - b*z
    
    return f_n, xz

def step(X_n, f_nm1):
    
    # Derivatives of the X, Y, Z state
    f_n = rhs(X_n)

    # Adams Bashforth
    X_np1 = X_n + dt*(3.0/2.0*f_n - 0.5*f_nm1)
    
    # Euler
    # X_np1 = X_n + dt*f_n
    
    return X_np1, f_n

def step_with_surrogate(X_n, f_nm1):

    # Derivatives of the X, Y, Z state
    f_n, xz = rhs_surrogate(X_n)

    # Adams Bashforth
    X_np1 = X_n + dt*(3.0/2.0*f_n - 0.5*f_nm1)
    
    # Euler
    # X_np1 = X_n + dt*f_n
   
    return X_np1, f_n, xz

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

X_train = (X[:, 0:2]).reshape([n_steps, 2])
y_train = (X[:, 0]*X[:, 2]).reshape([n_steps, 1])
n_train = X_train.shape[0]
n_feat = X_train.shape[1]
n_out = y_train.shape[1]

feat_eng = es.methods.Feature_Engineering()
n_bins = 20
feat_eng.bin_data(y_train, n_bins = n_bins)
sampler = es.methods.SimpleBin(feat_eng)

load = False

if not load:
    surrogate = es.methods.RNN(X_train, feat_eng.y_idx_binned, alpha = 0.001, loss = 'cross_entropy',
                               decay_rate = 0.9, decay_step = 10**4, activation = 'tanh',
                               bias = True, n_neurons = 32, n_layers = 2, sequence_size = 500,
                               n_out = n_bins, n_softmax = 1, training_mode='offline',
                               save = False, param_specific_learn_rate = True,
                               standardize_y = False)
    
    surrogate.train(5000)
else:
    surrogate = es.methods.RNN(X_train, y_train)
    surrogate.load_ANN()

mean_feat = surrogate.X_mean
std_feat = surrogate.X_std
# mean_data = surrogate.y_mean
# std_data = surrogate.y_std

test = []
surrogate.window_idx = 0
S = X_train.shape[0]

X_test = (X_train - surrogate.X_mean)/surrogate.X_std
surrogate.clear_history()
# surrogate.training_mode = 'offline'

I = 1000
for i in range(I):
    o_i, idx_max, idx_rvs = surrogate.get_softmax(X_test[i].reshape([1, n_feat]))
    test.append(sampler.resample(idx_max))
# test = list(chain(*test))

plt.plot(test)
plt.plot(y_train[0:I], 'ro')

X_n = X[0:2, 0]
f_nm1 = X_dot[0:2, 0]

n_pred = n_train
X_surr = np.zeros([n_pred, 2])
X_surr_dot = np.zeros([n_pred, 2])
outputs = np.zeros([n_pred, n_out])

for n in range(n_pred):
    
    X_surr[n, :] = X_n
        
    #step in time
    X_np1, f_n, xz = step_with_surrogate(X_n, f_nm1)

    X_surr_dot[n, :] = f_n
    outputs[n, :] = xz

    #update variables
    X_n = X_np1
    f_nm1 = f_n
    
plt.figure()
plt.plot(X_surr[:, 0], X_surr[:, 1], 'b+')

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