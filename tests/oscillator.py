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

plt.close('all')

dt = 0.01
t_end = 25.0
t = np.arange(0.0, t_end, dt)

store = True

x_n = 2.0
y_n = 0.0
f_nm1 = rhs_X(x_n, y_n)
g_nm1 = rhs_Y(x_n, y_n)

sol = np.zeros([t.size, 2])
sol[0, 0] = x_n
sol[0, 1] = y_n

#start time integration
idx = 1
for t_i in t[1:]:

    x_np1, y_np1, f_n, g_n = step(x_n, y_n, f_nm1, g_nm1)
    
    sol[idx, 0] = x_np1
    sol[idx, 1] = y_np1
    idx += 1
    
    x_n = x_np1
    y_n = y_np1
    f_nm1 = f_n
    g_nm1 = g_n    
    
#plot results
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sol[:, 0], sol[:, 1])

post_proc = es.methods.Post_Processing()

#store results
if store == True:
    #store results
    samples = {'x':sol[:, 0], 'y':sol[:, 1]}

    post_proc.store_samples_hdf5(samples)

plt.show()