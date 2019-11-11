def rhs_X(x, y):
    
    return -0.1*x**3 + 2.0*y**3    

def rhs_Y(x, y):
    
    return -2.0*x**3 - 0.1*y**3

def step(x_n, y_n, f_nm1, g_nm1):

    f_n = rhs_X(x_n, y_n)
    g_n = rhs_Y(x_n, y_n)    
    
    x_np1 = x_n + dt*(1.5*f_n - 0.5*f_nm1)
    y_np1 = y_n + dt*(1.5*g_n - 0.5*g_nm1)
    
    return x_np1, y_np1, f_n, g_n

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import easysurrogate as es
import h5py, os 
from itertools import chain

plt.close('all')

dt = 0.01
t_end = 25.0
t = np.arange(0.0, t_end, dt)

HOME = os.path.abspath(os.path.dirname(__file__))

#load training data
store_ID = 'oscillator'
QoI = ['x', 'y']
h5f = h5py.File(HOME + '/samples/' + store_ID + '.hdf5', 'r')

print('Loading', HOME + '/samples/' + store_ID + '.hdf5')

for q in QoI:
    print(q)
    vars()[q] = h5f[q][:]

state = np.zeros([x.size, 2])
state[:, 0] = x
state[:, 1] = y

feat_eng = es.methods.Feature_Engineering(state, state)

#lags = [[1]]
#max_lag = np.max(list(chain(*lags)))
#
#X_train, y_train = feat_eng.lag_training_data([state], lags = lags)

X_train = state[0:-1, :]
y_train = state[1:, :] - state[0:-1, :]

train = True
if train:
    
#    #standard regression neural net
#    surrogate = es.methods.ANN(X=X_train, y=y_train, n_layers=2, n_neurons=256, n_out=2,
#                               activation='hard_tanh', batch_size=32,
#                               lamb=0.01, decay_step=10**5, decay_rate=0.9, standardize_X=False,
#                               standardize_y=False, save=False)

    #physics-informed neural net
    surrogate = es.methods.ANN(X=X_train, y=y_train, n_layers=2, n_neurons=256, n_out=2, alpha=0.001,
                               activation='hard_tanh', batch_size=32, loss='user_def_squared',
                               lamb=0.0, decay_step=10**4, decay_rate=0.9, standardize_X=False,
                               standardize_y=False, save=False, bias=True)

    surrogate.train(10000, store_loss=True)
else:    
    surrogate = es.methods.ANN(X=X_train, y=y_train)
    surrogate.load_ANN()

n_steps = X_train.shape[0]
sol = np.zeros([n_steps, 2])
sol[0, :] = X_train[0]

state_n = X_train[1]
f_nm1 = surrogate.feed_forward(X_train[0].reshape([1,2])).flatten()

for i in range(1, n_steps):
    #standard ann pred
#    state_np1 = surrogate.feed_forward(state_n.reshape([1,2])).flatten()
    
    f_n = surrogate.feed_forward(state_n.reshape([1,2])).flatten()
        
    state_np1 = state_n + dt*(1.5*f_n - 0.5*f_nm1)
    
    sol[i,:] = state_np1
    
    f_nm1 = f_n
    state_n = state_np1
    
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sol[:, 0], sol[:, 1], label='PINN')
ax.plot(state[:, 0], state[:, 1], '--k', label='reference')
leg = plt.legend(loc=0)

plt.show()