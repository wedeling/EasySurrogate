"""
===============================================================================
Applies the Kernel Mixture Network to the Lorenz 96 system
-------------------------------------------------------------------------------
Reference: 
    
Luca Ambrogioni et al., "The Kernel Mixture Network: A Nonparametric Method 
for Conditional Density Estimation of Continuous Random Variables"

https://arxiv.org/abs/1705.07111

===============================================================================
"""

#########################
# Lorenz 96 subroutines #
#########################

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

#####################
# Other subroutines #
#####################          

def animate(i):
    """
    Generate a movie frame
    """
    
    plt1 = ax1.plot(dom_full, pdf_full, '--k', label=r'full pdf')
    plt2 = ax1.plot(dom, kde[:,0]/np.max(kde[:, 0]), 'b', label=r'conditional pdf')
    plt3 = ax1.plot(y_train[i][0], 0.0, 'ro', label=r'data')
    plt4 = ax2.plot(t[0:i], y_train[0:i, 0], 'ro', label=r'data')
    plt5 = ax2.plot(t[0:i], samples[0:i, 0], 'g', label='random sample')

    if i == 0:
        ax1.legend(loc=1, fontsize=9)
        ax2.legend(loc=1, fontsize=9)

    ims.append((plt1[0], plt2[0], plt3[0], plt4[0], plt5[0],))

def compute_kde(dom, w):
    
    """
    Compute a kernel density estimate (KDE) given the weights w
    
    Parameters:
        - dom (array): domain of the KDE
        - w (array): weights of the KDE, as predicted by the network. Sum to 1.
    """
    
    K = norm.pdf(dom, mu, sigma)
    w = w.reshape([w.size, 1])
    
    return np.sum(w*K, axis=0)

###############
# Main program
###############
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import easysurrogate as es
import h5py, os
from itertools import chain, product
from scipy.stats import norm, rv_discrete

plt.close('all')

#####################
#Lorenz96 parameters
#####################
K = 18
J = 20
F = 10.0
h_x = -1.0
h_y = 1.0
epsilon = 0.5

##################
# Time parameters
##################
dt = 0.01
t_end = 1000.0
t = np.arange(0.0, t_end, dt)
#time lags per feature
lags = [[1]]
max_lag = np.max(list(chain(*lags)))

###################
# Simulation flags
###################
train = True       #train the network
make_movie = True   #make a movie (of the training)
predict = True      #predict using the learned SGS term
store = True        #store the prediction results

#####################
# Network parameters
#####################

#Feature engineering object - loads data file
feat_eng = es.methods.Feature_Engineering()
#get training data
h5f = feat_eng.get_hdf5_file()

#Large-scale and SGS data - convert to numpy array via [()]
X_data = h5f['X_data'][()]
B_data = h5f['B_data'][()]

#Lag features as defined in 'lags'
X_train, y_train = feat_eng.lag_training_data([X_data], B_data, lags = lags)

#number of kernel means along the domain
n_means = 20
#kernel means
mu = np.linspace(np.min(y_train), np.max(y_train), n_means+1)
#kernel stdevs
sigma = np.linspace(0.2, 0.3, 3)
#tensor product of all kernel means and stdevs
#NOTE: ASSUMING HERE THAT ALL OUTPUTS HAVE THE SAME DOMAIN.
#      THIS IS TRUE FOR L96, NOT IN GENERAL
kernel_props = np.array(list(chain(product(mu, sigma))))
#final kernel means and stdevs
mu = kernel_props[:, 0]
sigma = kernel_props[:, 1]
mu = mu.reshape([mu.size, 1])
sigma = sigma.reshape([sigma.size, 1])

#number of softmax layers (one per output)
n_softmax = K

#number of output neurons 
n_out = mu.size*n_softmax

#train the neural network
if train:
    surrogate = es.methods.ANN(X=X_train, y=y_train, n_layers=3, n_neurons=256, 
                               n_out=n_out, loss='kvm', bias = True,
                               activation='tanh', activation_out='linear', 
                               n_softmax=n_softmax, batch_size=512,
                               lamb=0.0, decay_step=10**4, decay_rate=0.9, alpha=0.001,
                               standardize_X=False, standardize_y=False, save=True,
                               kernel_means = mu, kernel_stds = sigma)

    print('===============================')
    print('Training Kernel Mixture Network...')
  
    #train network for N_inter mini batches
    N_iter = 10000
    surrogate.train(N_iter, store_loss = True)
    
#load a neural network from disk
else:
    #first create dummy ANN object    
    surrogate = es.methods.ANN(X=X_train, y=y_train)
    #the load trained network from disk
    surrogate.load_ANN()

#number of features
n_feat = surrogate.n_in
    
#Post processing object
post_proc = es.methods.Post_Processing()

#if True make a movie of the solution
if make_movie:
    
    print('===============================')
    print('Making movie...')

    #pdf of SGS
    dom_full, pdf_full = post_proc.get_pde(y_train[0:-1:10])
    
    ims = []
    fig = plt.figure(figsize=[8,4])
    ax1 = fig.add_subplot(121, xlabel=r'$B_k$', ylabel=r'', yticks=[])
    ax2 = fig.add_subplot(122, xlabel=r'time', ylabel=r'$B_k$')
    plt.tight_layout()

    #number of time steps to use in movie
    n_movie = 500

    #number of points to use for plotting the conditional pdfs
    n_kde = 100
    
    #domain of the conditional pdfs
    dom = np.linspace(np.min(B_data), np.max(B_data), n_kde)
    
    #allocate memory
    kde = np.zeros([n_kde, n_softmax])    
    samples = np.zeros([n_movie, n_softmax])
    
    #make movie by evaluating the network at TRAINING inputs
    for i in range(n_movie):
        
        #draw a random sample from the network
        w, _, idx = surrogate.get_softmax(X_train[i].reshape([1, n_feat]))
        for j in range(n_softmax):
            kde[:, j] = compute_kde(dom, w[j])
            samples[i, j] = norm.rvs(mu[idx[j]], sigma[idx[j]])
        if np.mod(i, 100) == 0:
            print('i =', i, 'of', n_movie)
        
        #create a single frame, store in 'ims'
        animate(i)

    #make a movie of all frame in 'ims'
    im_ani = animation.ArtistAnimation(fig, ims, interval=20, 
                                       repeat_delay=2000, blit=True)
    im_ani.save('./movies/kvm.mp4')
    
    print('done')

##################################
# make predictions, with ANN SGS #
##################################

if predict:
    
    print('===============================')
    print('Predicting with stochastic SGS...')
    
    #features are time lagged, use the data to create initial feature set
    for i in range(max_lag):
        feat_eng.append_feat([X_data[i]], max_lag)
    
    #initial conditions
    X_n = X_data[max_lag]
    B_n = B_data[max_lag]
    
    X_nm1 = X_data[max_lag-1]
    B_nm1 = B_data[max_lag-1]
    
    #initial right-hand sides
    f_nm1 = rhs_X(X_nm1, B_nm1)
    
    #allocate memory for solutions
    X_ann = np.zeros([t.size, K])
    B_ann = np.zeros([t.size, K])
    
    #start time integration
    idx = 0
    for t_i in t[max_lag:]:
    
        #get time lagged features from Feature Engineering object
        feat = feat_eng.get_feat_history()

        #SGS solve, draw random sample from network
        _, _, rvs = surrogate.get_softmax(feat.reshape([1, n_feat]))
        B_n = norm.rvs(mu[rvs], sigma[rvs])  
        B_n = B_n.flatten()
        
        #solve large-scale equation
        X_np1, f_n = step_X(X_n, f_nm1, B_n)
        
        #append the features to the Feature Engineering object
        feat_eng.append_feat([X_np1], max_lag)
    
        #store solutions
        X_ann[idx, :] = X_n
        B_ann[idx, :] = B_n
        
        idx += 1
    
        #update variables
        X_n = X_np1
        f_nm1 = f_n
        
        if np.mod(idx, 1000) == 0:
            print('t =', np.around(t_i, 1), 'of', t_end)
       
    print('done')

    #############   
    # Plot PDEs #
    #############

    print('===============================')
    print('Postprocessing results')
       
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel=r'$X_k$')
    
    post_proc = es.methods.Post_Processing()
    X_dom_surr, X_pde_surr = post_proc.get_pde(X_ann.flatten()[0:-1:10])
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
    R_sol = post_proc.auto_correlation_function(X_ann[:, 0], max_lag=1000)
    
    dom_acf = np.arange(R_data.size)*dt
    
    ax.plot(dom_acf, R_data, 'ko', label='L96')
    ax.plot(dom_acf, R_sol, label='ANN')
    
    leg = plt.legend(loc=0)
    
    plt.tight_layout()
    
    #############   
    # Plot CCFs #
    #############
    
    fig = plt.figure()
    ax = fig.add_subplot(111, ylabel='CCF', xlabel='time')
    
    C_data = post_proc.cross_correlation_function(X_data[:,0], X_data[:,1], max_lag=1000)
    C_sol = post_proc.cross_correlation_function(X_ann[:, 0], X_ann[:, 1], max_lag=1000)
    
    dom_ccf = np.arange(C_data.size)*dt
    
    ax.plot(dom_ccf, C_data, 'ko', label='L96')
    ax.plot(dom_ccf, C_sol, label='ANN')
    
    leg = plt.legend(loc=0)
    
    plt.tight_layout()
    
    #store simulation results
    if store:
        samples = {'X': X_ann, 'B':B_ann, \
                   'dom_acf':dom_acf, 'acf_data':R_data, 'acf_ann':R_sol, \
                   'dom_ccf':dom_ccf, 'ccf_data':C_data, 'ccf_ann':C_sol}
        post_proc.store_samples_hdf5(samples)

plt.show()