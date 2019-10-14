def init_movie():
    for line in lines:
        line.set_data([],[])
    return lines

def animate(s):

    print('Processing frame', s, 'of', sol.shape[0])
    #add the periodic BC to current sample to create a closed plot
#    line.set_data(theta, np.append(sol[s, 0:K], sol[s, 0:K]))    
#    line_Y.set_data(theta_Y, np.append(sol[s, K:], sol[s, K]))
    
    xlist = [theta, theta_Y]
    ylist = [np.append(sol[s, 0:K], sol[s, 0]), np.append(sol[s, K:], sol[s, K])]

    #for index in range(0,1):
    for lnum,line in enumerate(lines):
        line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately. 

    return lines

#def lorenz96(X, t):
#    
#    rhs_X = np.zeros(K)
#    
#    #first treat boundary cases (k=1, k=2 and k=K)
#    rhs_X[0] = -X[K-2]*X[K-1] + X[K-1]*X[1] - X[0] + F
#    rhs_X[1] = -X[K-1]*X[0] + X[0]*X[2] - X[1] + F
#    rhs_X[K-1] = -X[K-3]*X[K-2] + X[K-2]*X[0] - X[K-1] + F
#    
#    #treat interior points
#    for k in range(2, K-1):
#        rhs_X[k] = -X[k-2]*X[k-1] + X[k-1]*X[k+1] - X[k] + F
#        
#    return rhs_X

def lorenz96_2layer(U, t):
    """
    Lorenz96 two-layer model
    """
    
    X = np.array(U[0:K])
    Y = np.array(U[K:])    
    Y = Y.reshape([J, K])
    
    rhs_X = np.zeros(K)
    rhs_Y = np.zeros([J, K])
    
    #first treat boundary cases (k=1, k=2 and k=K)
    rhs_Y[:, 0] = rhs_Y_k(Y, X[0], 0)
    rhs_X[0] = -X[K-2]*X[K-1] + X[K-1]*X[1] - X[0] + F + h_x*np.mean(Y[:, 0])
    
    rhs_Y[:, 1] = rhs_Y_k(Y, X[1], 1)
    rhs_X[1] = -X[K-1]*X[0] + X[0]*X[2] - X[1] + F + h_x*np.mean(Y[:, 1])

    rhs_Y[:, K-1] = rhs_Y_k(Y, X[K-1], K-1)
    rhs_X[K-1] = -X[K-3]*X[K-2] + X[K-2]*X[0] - X[K-1] + F + h_x*np.mean(Y[:, K-1])
    
    #treat interior points
    for k in range(2, K-1):
        rhs_Y[:, k] = rhs_Y_k(Y, X[k], k)
        rhs_X[k] = -X[k-2]*X[k-1] + X[k-1]*X[k+1] - X[k] + F + h_x*np.mean(Y[:, k])
        
    return list(chain(*[rhs_X, rhs_Y.flatten()]))

def rhs_Y_k(Y, X_k, k):
    """
    Compute the right-hand side of Y for fixed k
    """
    
    rhs_Yk = np.zeros(J)
    
    #first treat boundary cases (j=1, j=J-1, j=J-2)
    if k > 0:
        idx = k-1
    else:
        idx = K-1
    rhs_Yk[0] = (Y[1, k]*(Y[J-1, idx] - Y[2, k]) - Y[0, k] + h_y*X_k)/epsilon

    if k < K-1:
        idx = k+1
    else:
        idx = 0
    rhs_Yk[J-2] = (Y[J-1, k]*(Y[J-3, k] - Y[0, idx]) - Y[J-2, k] + h_y*X_k)/epsilon
    rhs_Yk[J-1] = (Y[0, idx]*(Y[J-2, k] - Y[1, idx]) - Y[J-1, k] + h_y*X_k)/epsilon

    #treat interior points
    for j in range(1, J-2):
        rhs_Yk[j] = (Y[j+1, k]*(Y[j-1, k] - Y[j+2, k]) - Y[j, k] + h_y*X_k)/epsilon
        
    return rhs_Yk

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from itertools import chain

plt.close('all')

#Lorenz96 parameters
K = 18
J = 20
F = 10.0
h_x = -1.0
h_y = 1.0
epsilon = 0.5 

#equilibrium initial condition for X, zero IC for Y
X = np.ones(K)*F
X[10] += 0.01 # add small perturbation to 10th variable
Y = np.zeros(J*K)

#initial condition
U_init = list(chain(*[X, Y]))

#time param
t_end = 10.0
dt = 0.01
t = np.arange(0.0, t_end, dt)

#solve system
sol = odeint(lorenz96_2layer, U_init, t)

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
theta = np.linspace(0.0, 2.0*np.pi, K+1)
theta_Y = np.linspace(0.0, 2.0*np.pi, J*K+1)
#create a while in the middle to the plot, scales plot better
ax.set_rorigin(-22)
#remove set radial tickmarks, no labels
ax.set_rgrids([-10, 0, 10], labels=['', '', ''])[0][1]
ax.legend(loc=1)

#if True make a movie of sthe solution, if not just plot final solution
make_movie = True
if make_movie:

    lines = []
    legends = ['X', 'y']
    for i in range(2):
        lobj = ax.plot([],[],lw=2, label=legends[i])[0]
        lines.append(lobj)
    
    anim = FuncAnimation(fig, animate, init_func=init_movie,
                         frames=np.arange(0, sol.shape[0], 10), blit=True)    
    anim.save('demo.gif', writer='imagemagick')
else:
    ax.plot(theta, np.append(sol[-1, 0:K], sol[-1, 0]), label='X')    
    ax.plot(theta_Y, np.append(sol[-1, K:], sol[-1, K]), label='y')    

plt.show()