def step(p_n, q_n):
    
    p_np1 = p_n - dt*q_n
    q_np1 = q_n + dt*p_np1
    
    #Hamiltonian
    H_n = 0.5*p_n**2 + 0.5*q_n**2
    
    return p_np1, q_np1, H_n

import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es

plt.close('all')

##################
# Initialisation #
##################

# distinguished particle:
q_n = 1  # position
p_n = 0  # momentum

#####################################
# Integration with symplectic Euler #
#####################################

dt = 0.01  # integration time step
M = 10**4

#####################
# Network parameters
#####################

#Feature engineering object - loads data file
feat_eng = es.methods.Feature_Engineering()
#get training data
h5f = feat_eng.get_hdf5_file()

p_n = h5f['p_n'][()].flatten()
q_n = h5f['q_n'][()].flatten()
dpdt = h5f['dpdt'][()].flatten()
dqdt = h5f['dqdt'][()].flatten()

X_train = np.zeros([M, 2])
X_train[:, 0] = p_n
X_train[:, 1] = q_n

y_train = np.zeros([M, 2])
y_train[:, 0] = dpdt
y_train[:, 1] = dqdt

surrogate = es.methods.ANN(X=X_train, y=y_train, n_layers=3, 
                           n_neurons = 200, alpha = 0.001, 
                           n_softmax = 0, n_out = 2, 
                           loss = 'squared', activation='hard_tanh', batch_size=512,
                           lamb=0.0, decay_step=10**4, decay_rate=0.9, 
                           standardize_X=False, standardize_y=False, save=False)

#train network for N_inter mini batches
N_iter = 2000
surrogate.train(N_iter, store_loss = True, sequential = False)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(p_n, q_n, 'ro')

y = np.zeros([M, 2])

p_n = X_train[0][0]
q_n = X_train[0][1]

for i in range(M):
    
    feat = np.array([p_n, q_n])
    
    diff = surrogate.feed_forward(feat.reshape([1, 2])).flatten()
    
    p_n = p_n + diff[0]*dt
    q_n = q_n + diff[1]*dt

    y[i, 0] = p_n; y[i, 1] = q_n
    
ax.plot(y[:, 0], y[:, 1], 'b+')

plt.axis('equal')
plt.tight_layout()

plt.show()