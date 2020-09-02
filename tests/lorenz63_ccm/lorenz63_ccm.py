def rhs(X_n, s=10, r=28, b=8/3):

    x = X_n[0]; y = X_n[1]; z = X_n[2]
    
    f_n = np.zeros(3)
    
    f_n[0] = s*(y - x)
    f_n[1] = r*x - y - x*z
    f_n[2] = x*y - b*z
    
    return f_n

def step(X_n):

    ##############################
    # Easysurrogate modification #
    ##############################    

    # Remove call to the 'small' scale equation
    # Derivatives of the X, Y, Z state
    # f_n = rhs(X_n)

    # Replace with call to surrogate
    Y_n = campaign.surrogate.predict(X_n, stochastic=False)
    f_n = 10*(Y_n - X_n)

    ##################################
    # End Easysurrogate modification #
    ##################################

    # Euler
    X_np1 = X_n + dt*f_n

    return X_np1, Y_n

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

#initial condition right-hand side
f_nm1 = rhs(X_n)
X = np.zeros([n_steps, 3])

##############################
# Easysurrogate modification #
##############################

import easysurrogate as es

#load pre-trained campaign
campaign = es.Campaign(load_state=True)

#change IC
data_frame = campaign.load_hdf5_data()
X_n = data_frame['X'][campaign.surrogate.max_lag]
X = np.zeros([n_steps, 2])

##################################
# End Easysurrogate modification #
##################################

for n in range(n_steps):

    #step in time
    X_np1, Y_n = step(X_n)

    #update variables
    X_n = X_np1

    X[n, 0] = X_n
    X[n, 1] = Y_n

#use easysurrogate to store the data
campaign.store_data_to_hdf5({'X': X[:, 0], 'Y': X[:, 1]})

plt.show()