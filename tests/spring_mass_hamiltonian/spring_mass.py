def step(p_n, q_n):
    
    p_np1 = p_n - dt*q_n
    q_np1 = q_n + dt*p_np1
    
    #Hamiltonian
    H_n = 0.5*p_n**2 + 0.5*q_n**2
    
    return p_np1, q_np1, H_n

import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es

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

# data to store
QoI = ['q_n', 'p_n', 'H_n', 'dqdt', 'dpdt']
Size = [1, 1, 1, 1, 1]
data = {}
idx = 0
for qoi in QoI:
    data[qoi] = np.zeros([M, Size[idx]])
    idx += 1

# initial Nskip*10^3 integration steps are discarded as transient
for j in range(1000):
    p_np1, q_np1, H_n = step(p_n, q_n)

    p_n = p_np1
    q_n = q_np1

j = 0
for i in range(M):
    p_np1, q_np1, H_n = step(p_n, q_n)

    dpdt = (p_np1 - p_n)/dt
    dqdt = (q_np1 - q_n)/dt
    
    idx += 1

    for qoi in QoI:
        data[qoi][j,:] = eval(qoi)
    j += 1

    p_n = p_np1
    q_n = q_np1
    
post_proc = es.methods.Post_Processing()
post_proc.store_samples_hdf5(data)

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='q', ylabel='p')

ax.plot(data['q_n'], data['p_n'])

plt.axis('equal')
plt.tight_layout()

plt.show()