def rhs(X_n, s=10, r=28, b=8/3):

    x = X_n[0]; y = X_n[1]; z = X_n[2]
    
    f_n = np.zeros(3)
    
    f_n[0] = s*(y - x)
    f_n[1] = r*x - y - x*z
    f_n[2] = x*y - b*z
    
    return f_n

def step(X_n):
    
    # Derivatives of the X, Y, Z state
    f_n = rhs(X_n)
  
    # Euler
    X_np1 = X_n + dt*f_n
    
    return X_np1

def plot_lorenz(xs, ys, zs, title='Lorenz63'):

    fig = plt.figure(title)
    ax = fig.gca(projection='3d')
    
    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(title)
    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')

n_steps = 25000
dt = 0.01

#initial condition
X_n = np.zeros(3)
X_n[0] = 0.0; X_n[1]  = 1.0; X_n[2] = 1.05

X = np.zeros([n_steps, 3])

for n in range(n_steps):
    
    #step in time
    X_np1 = step(X_n)

    #update variables
    X_n = X_np1

    X[n, :] = X_n

#use easysurrogate to store the data
import easysurrogate as es
campaign = es.Campaign()
campaign.store_data_to_hdf5({'X':X[:, 0], 'Y':X[:, 1], 'Z':X[:, 2]})

plot_lorenz(X[:, 0], X[:, 1], X[:, 2])
plt.show()