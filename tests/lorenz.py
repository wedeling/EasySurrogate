def get_data(stepCnt, reference = True):
    """
    Get data for Lorenz 63 system
    """
    dt = 0.01
    
    # Need one more for the initial values
    xs = np.empty((stepCnt + 1,))
    ys = np.empty((stepCnt + 1,))
    zs = np.empty((stepCnt + 1,))
    
    # Setting initial values
    xs[0], ys[0], zs[0] = (0., 1., 1.05)
    
    # Stepping through "time".
    for i in range(stepCnt):
        # Derivatives of the X, Y, Z state
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        
        if reference:
            zs[i + 1] = zs[i] + (z_dot * dt)
        else:
            covar = np.array([xs[i], ys[i]])
            zs[i + 1] = surrogate.sample(covar.reshape([1,2]))
        
    return xs, ys, zs

def lorenz(x, y, z, s=10, r=28, b=2.667):
    """
    Lorenz 1963 Deterministic Nonperiodic Flow
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def plot_lorenz(xs, ys, zs):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import easysurrogate as es

plt.close('all')

stepCnt = 10000

xs, ys, zs = get_data(stepCnt)
plot_lorenz(xs, ys, zs)

covar = np.array([xs, ys]).T
surrogate = es.methods.Resampler(covar, zs, 1, 30, [1, 1])

xs, ys, zs = get_data(stepCnt, reference = False)
plot_lorenz(xs, ys, zs)

plt.show()