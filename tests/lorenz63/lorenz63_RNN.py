def rhs(X_n, s, r=28, b=2.667):
    """
    Lorenz 1963 Deterministic Nonperiodic Flow
    """
    x = X_n[0]; y = X_n[1]; z = X_n[2]
    
    f_n = np.zeros(3)
    
    f_n[0] = s*(y - x)
    f_n[1] = r*x - y - x*z
    f_n[2] = x*y - b*z
    
    return f_n

def rhs_surrogate(X_n, Y_n, s):

    feat = np.array([X_n]).flatten()
    feat = (feat - mean_feat)/std_feat
    y = surrogate.feed_forward(feat.reshape([1, n_feat]))[0]
    y = y*std_data + mean_data
    
    # y = Y_n + y_dot*dt

    f_n = s*(y - X_n)
    # f_n[1] = r*x - y - x*z
    # z_dot = x*y - b*z
    
    return f_n, y

def step(X_n, f_nm1):
    
    # Derivatives of the X, Y, Z state
    f_n = rhs(X_n, sigma)

    # Adams Bashforth
    # X_np1 = X_n + dt*(3.0/2.0*f_n - 0.5*f_nm1)
    
    # Euler
    X_np1 = X_n + dt*f_n
    
    return X_np1, f_n

def step_with_surrogate(X_n, Y_n, f_nm1):

    # Derivatives of the X, Y, Z state
    f_n, Y_np1 = rhs_surrogate(X_n, Y_n, sigma)

    # Adams Bashforth
    # X_np1 = X_n + dt*(3.0/2.0*f_n - 0.5*f_nm1)
    
    # Euler
    X_np1 = X_n + dt*f_n
   
    return X_np1, Y_np1, f_n

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

n_steps = 10**4
dt = 0.01
sigma = 10.0
alpha = 1.0-sigma*dt

X = np.zeros(n_steps); Y = np.zeros(n_steps); Z = np.zeros(n_steps) 
X_dot = np.zeros(n_steps); Y_dot = np.zeros(n_steps); Z_dot = np.zeros(n_steps) 

#initial condition
X_n = np.zeros(3)
X_n[0] = 0.0; X_n[1]  = 1.0; X_n[2] = 1.05

#initial condition right-hand side
f_nm1 = rhs(X_n, sigma)

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
    

X_train = X[:, 0:1].reshape([n_steps, 1])
y_train = X[:, 1].reshape([n_steps, 1])
n_train = X_train.shape[0]
n_feat = X_train.shape[1]

X_train = np.linspace(0, 2*np.pi, 1000).reshape([1000, 1])
y_train = np.sin(X_train).reshape([1000, 1])

surrogate = es.methods.RNN(X_train, y_train, alpha = 0.001,
                           decay_rate = 0.9, decay_step = 10**5, activation = 'tanh',
                           bias = True, n_neurons = 16, n_layers = 2, sequence_size = 100,
                           n_out = y_train.shape[1], training_mode='offline', increment=1,
                           save = False, param_specific_learn_rate = True)

surrogate.train(10000)

mean_feat = surrogate.X_mean
std_feat = surrogate.X_std
mean_data = surrogate.y_mean
std_data = surrogate.y_std

test = []
surrogate.window_idx = 0
S = X_train.shape[0]

X_test = (X_train - surrogate.X_mean)/surrogate.X_std
# surrogate.clear_history()

n_surr = 1000

# for i in range(n_surr):
#     test.append(surrogate.feed_forward(back_prop = False)[-1][0])
# # test = list(chain(*test))

#offline prediction one step ahead
for i in range(100, n_surr):
    x = np.array([X_test[i]])
    test.append(surrogate.feed_forward(x_sequence = x)[0][0])

plt.plot(test)
# plt.plot(surrogate.y[0:n_surr], 'ro')

"""
X_surr = np.zeros([n_steps, 1])
X_surr_dot = np.zeros([n_steps, 1])

X_n = X[1, 0]
Y_n = X[1, 1]
f_nm1 = X_dot[0, 0]

inputs = []; outputs = []

for n in range(n_train):
    
    X_surr[n, :] = X_n
        
    #step in time
    X_np1, Y_np1, f_n = step_with_surrogate(X_n, Y_n, f_nm1)

    inputs.append(alpha*X_n + (1-alpha)*Y_n)
    outputs.append(Y_np1[0])

    X_surr_dot[n, :] = f_n

    #update variables
    X_n = X_np1
    Y_n = Y_np1
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
"""
plt.show()