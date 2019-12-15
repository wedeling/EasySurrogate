def animate(s):

    print('Processing frame', s)
#    xlist = [dom, y_train[s], dom_full, np.arange(s), np.arange(s)]
#    ylist = [kde[s, :]/np.max(kde[s,:]), 0.0, pdf_full, y_train[0:s], samples[0:s]]

#    for lnum,line in enumerate(lines):
#        line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately. 

    I = 2
    for i in range(I):
        lines[2*i].set_data(dom, kde[s, :, i]/np.max(kde[s, :, i]))
        lines[2*i + 1].set_data(y_train[s, i], 0.0)
        
    lines[2*I].set_data(dom_full, pdf_full)
    lines[2*I+1].set_data(np.arange(s), y_train[0:s, 0])
    lines[2*I+2].set_data(np.arange(s), samples[0:s, 0])

    ax2.relim()            # reset intern limits of the current axes
    ax2.autoscale_view()   # reset axes limits

    ax2.legend(loc=1)
        
    return lines

def rhs_X(X, B):
    """
    Compute the right-hand side of X
    
    Parameters:
        - X (array, size K): large scale variables
        - B (array, size K): SGS term
        
    returns:
        - rhs_X (array, size K): right-hand side of X
    """
    
    rhs_X = np.zeros(K)
    
    #first treat boundary cases (k=1, k=2 and k=K)
    rhs_X[0] = -X[K-2]*X[K-1] + X[K-1]*X[1] - X[0] + F
    
    rhs_X[1] = -X[K-1]*X[0] + X[0]*X[2] - X[1] + F
    
    rhs_X[K-1] = -X[K-3]*X[K-2] + X[K-2]*X[0] - X[K-1] + F
    
    #treat interior points
    for k in range(2, K-1):
        rhs_X[k] = -X[k-2]*X[k-1] + X[k-1]*X[k+1] - X[k] + F
        
    rhs_X += B
        
    return rhs_X

def rhs_Y_k(Y, X_k, k):
    """
    Compute the right-hand side of Y for fixed k
    
    Parameters:
        - Y (array, size (J,K)): small scale variables
        - X_k (float): large scale variable X_k
        - k (int): index k
        
    returns:
        - rhs_Yk (array, size J): right-hand side of Y for fixed k
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

def step_X(X_n, f_nm1, B):
    """
    Time step for X equation, using Adams-Bashforth
    
    Parameters:
        - X_n (array, size K): large scale variables at time n
        - f_nm1 (array, size K): right-hand side of X at time n-1
        - B (array, size K): SGS term
        
    Returns:
        - X_np1 (array, size K): large scale variables at time n+1
        - f_nm1 (array, size K): right-hand side of X at time n
    """
    
    #right-hand side at time n
    f_n = rhs_X(X_n, B)
    
    #adams bashforth
    X_np1 = X_n + dt*(3.0/2.0*f_n - 0.5*f_nm1)
    
    return X_np1, f_n

def step_Y(Y_n, g_nm1, X_n):
    """
    Time step for Y equation, using Adams-Bashforth
    
    Parameters:
        - Y_n (array, size (J,K)): small scale variables at time n
        - g_nm1 (array, size (J, K)): right-hand side of Y at time n-1
        - X_n (array, size K): large scale variables at time n
        
    Returns:
        - Y_np1 (array, size (J,K)): small scale variables at time n+1
        - g_nm1 (array, size (J,K)): right-hand side of Y at time n
    """

    g_n = np.zeros([J, K])
    for k in range(K):
        g_n[:, k] = rhs_Y_k(Y_n, X_n[k], k)
        
    Y_np1 = Y_n + dt*(3.0/2.0*g_n - 0.5*g_nm1)
    
    return Y_np1, g_n

def wave_variance(X):
    
    X_hat = np.fft.rfft(X, axis=0, norm='ortho')
    X_hat_mean = np.mean(X_hat, axis=0)
    X_hat_var = np.mean(np.abs(X_hat - X_hat_mean)**2, axis=0)
    
    return X_hat_var

def compute_kde(dom, w):
    
    K = norm.pdf(dom, mu, sigma)
    w = w.reshape([w.size, 1])
    
    return np.sum(w*K, axis=0)#/np.sum(w)  

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import easysurrogate as es
import h5py, os
from itertools import chain, product
from scipy.stats import norm, rv_discrete

plt.close('all')

#Lorenz96 parameters
K = 18
J = 20
F = 10.0
h_x = -1.0
h_y = 1.0
epsilon = 0.5

##trimodal Lorenz96 parameters
#K = 32
#J = 16
#F = 18.0
#h_x = -3.2
#h_y = 1.0
#epsilon = 0.5

dt = 0.01
t_end = 1000.0
t = np.arange(0.0, t_end, dt)

make_movie = True
store = False
train = True

# HOME = os.path.abspath(os.path.dirname(__file__))

# #load training data
# store_ID = 'L96'
# QoI = ['X_data', 'B_data', 'Y_data']
# h5f = h5py.File(HOME + '/samples/' + store_ID + '.hdf5', 'r')

# print('Loading', HOME + '/samples/' + store_ID + '.hdf5')

# for q in QoI:
#     print(q)
#     vars()[q] = h5f[q][:]

lags = [[1]]
max_lag = np.max(list(chain(*lags)))

data_eng = es.methods.Data_Engineering()
data_eng.set_training_data(['X_data'], ['B_data'])
X_train, y_train = data_eng.set_lagged_training_data(data_eng.X, lags = lags)
n_bins = 10
data_eng.bin_data(y_train, n_bins)

#anchor points kernels
x_p = np.linspace(np.min(y_train), np.max(y_train), n_bins+1)
sigma_j = np.linspace(0.2, 0.3, 3)
kernel_props = np.array(list(chain(product(x_p, sigma_j))))

mu = kernel_props[:, 0]
sigma = kernel_props[:, 1]
mu = mu.reshape([mu.size, 1])
sigma = sigma.reshape([sigma.size, 1])

n_softmax = K
n_bins = kernel_props.shape[0]

#if n_softmax > 1:
#    mu = np.tile(mu, n_softmax)
#    sigma = np.tile(sigma, n_softmax)

if train:
    surrogate = es.methods.ANN(X=X_train, y=data_eng.binned_data, n_layers=3, n_neurons=256, 
                              n_out=kernel_props.shape[0]*n_softmax, loss='kvm', bias = True,
                              activation='relu', activation_out='linear', n_softmax=n_softmax,
                              batch_size=512,
                              lamb=0.0, decay_step=10**4, decay_rate=0.9, alpha=0.001,
                              standardize_X=False, standardize_y=False, save=True,
                              kernel_means = mu, kernel_stds = sigma)
    surrogate.train(10000, store_loss = True)

else:    
    surrogate = es.methods.ANN(X=X_train, y=y_train)
    surrogate.load_ANN()

#if True make a movie of the solution, if not just plot final solution
if make_movie:
    
    post_proc = es.methods.Post_Processing()
    dom_full, pdf_full = post_proc.get_pde(y_train[0:-1:10])
    
    n_kde = 100
    n_train = np.int(0.1*X_train.shape[0])
    dom = np.linspace(np.min(B_data), np.max(B_data), n_kde)
    n_feat = surrogate.n_in
    kde = np.zeros([X_train.shape[0], n_kde, n_softmax])    
    samples = np.zeros([n_train, n_softmax])
    
    for i in range(n_train):
        #w = surrogate.feed_forward(X_train[i].reshape([1, n_feat]))
        w, _, idx = surrogate.get_softmax(X_train[i].reshape([1, n_feat]))
        for j in range(n_softmax):
            kde[i, :, j] = compute_kde(dom, w[j])
            samples[i, j] = norm.rvs(mu[idx[j]], sigma[idx[j]])
        if np.mod(i, 1000) == 0:
            print('i =', i, 'of', n_train)
    
    fig = plt.figure(figsize=[10,5])
    ax = fig.add_subplot(121)
    plt.xlim([dom[0], dom[-1]])
    plt.yticks([])
    plt.xlabel(r'$B_k$')
     
    ax2 = fig.add_subplot(122)
        
    plt.tight_layout()
 
    lines = []
#    symbols = ['-b', 'ro', '--k']
#    labels = [r'$\mathrm{Kernel\;Mixture\;Network}\;p\left(B_k|{\bf X}\right)$', 
#              r'$\mathrm{SGS\;data}\;B_k$', '']
    for i in range(2):
        lobj = ax.plot([], [], '-')[0]
        lines.append(lobj)
        lobj = ax.plot([], [], 'o')[0]
        lines.append(lobj)
        
    lobj = ax.plot([], [], '--k')[0]
    lines.append(lobj)

    symbols = ['ro', '-g']
    labels = [r'$\mathrm{SGS\;data}\;B_k$',
              r'$\mathrm{Random\;KVM\;sample}$']    
    for i in range(2):
        lobj = ax2.plot([], [], symbols[i], label=labels[i])[0]
        lines.append(lobj)
    
    # Set up formatting for the movie files
    anim = FuncAnimation(fig, animate, frames=np.arange(0, 500, 4))    
    Writer = writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)
    anim.save('demo.mp4', writer = writer)

for i in range(max_lag):
    data_eng.append_feat([X_data[i]], max_lag)
    
####################
# make predictions #
####################
    
n_feat = surrogate.n_in

X_n = X_data[max_lag]
B_n = B_data[max_lag]

X_nm1 = X_data[max_lag-1]
B_nm1 = B_data[max_lag-1]

#initial right-hand sides
f_nm1 = rhs_X(X_nm1, B_nm1)

#allocate memory for solutions
sol = np.zeros([t.size, K])

#start time integration
idx = 0
for t_i in t[max_lag:]:

    #ANN SGS solve
    feat = data_eng.get_feat_history()
    _, _, rvs = surrogate.get_softmax(feat.reshape([1, n_feat]))
    B_n = norm.rvs(mu[rvs], sigma[rvs])  
    B_n = B_n.flatten()
    
    #solve large-scale equation
    X_np1, f_n = step_X(X_n, f_nm1, B_n)
    data_eng.append_feat([X_np1], max_lag)

    #store solutions
    sol[idx, :] = X_n
    idx += 1

    #update variables
    X_n = X_np1
    f_nm1 = f_n
    
    if np.mod(idx, 1000) == 0:
        print('t =', np.around(t_i, 1), 'of', t_end)
   
#############   
# Plot PDEs #
#############
    
fig = plt.figure()
ax = fig.add_subplot(111, xlabel=r'$X_k$')

post_proc = es.methods.Post_Processing()
X_dom_surr, X_pde_surr = post_proc.get_pde(sol.flatten()[0:-1:10])
X_dom, X_pde = post_proc.get_pde(X_data.flatten()[0:-1:10])

ax.plot(X_dom, X_pde, 'ko', label='L96')
ax.plot(X_dom_surr, X_pde_surr, label='ANN')

plt.yticks([])

plt.legend(loc=0)

plt.tight_layout()

#############   
# Plot ACFs #
#############

fig = plt.figure()
ax = fig.add_subplot(111, ylabel='ACF', xlabel='time')

R_data = post_proc.auto_correlation_function(X_data[:,0], max_lag=1000)
R_sol = post_proc.auto_correlation_function(sol[:, 0], max_lag=1000)

dom = np.arange(R_data.size)*dt

ax.plot(dom, R_data, 'ko', label='L96')
ax.plot(dom, R_sol, label='ANN')

leg = plt.legend(loc=0)

plt.tight_layout()

#############   
# Plot CCFs #
#############

fig = plt.figure()
ax = fig.add_subplot(111, ylabel='CCF', xlabel='time')

C_data = post_proc.cross_correlation_function(X_data[:,0], X_data[:,1], max_lag=1000)
C_sol = post_proc.cross_correlation_function(sol[:, 0], sol[:, 1], max_lag=1000)

dom = np.arange(C_data.size)*dt

ax.plot(dom, C_data, 'ko', label='L96')
ax.plot(dom, C_sol, label='ANN')

leg = plt.legend(loc=0)

plt.tight_layout()

plt.show()