"""
===============================================================================
Applies the Quantized Softmax Network to the Lorenz 96 system
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
    Generate a movie frame for the training verification (neural net only)
    """
    
    if idx_max[0] == idx_data[0]:
        c = 'b'
    else:
        c = 'r'
    
    plt1 = ax1.vlines(np.arange(n_bins), ymin = np.zeros(n_bins), ymax = o_i[0],
                      colors = c, label=r'conditional pmf')
    plt2 = ax1.plot(idx_data[0], 0.0, 'ro', label=r'SGS data')
    plt3 = ax2.plot(t[0:i], y_train[0:i, 0], 'ro', alpha = 0.5, label=r'SGS data')
    plt4 = ax2.plot(t[0:i], samples[0:i, 0], 'b', label='random sample')

    if i == 0:
        ax1.legend(loc=1, fontsize=9)
        ax2.legend(loc=1, fontsize=9)

    ims.append((plt1, plt2[0], plt4[0], plt3[0],))
   
def animate_pred(i):
    """
    Generate a movie frame for the coupled system
    """
    plt1 = ax1.plot(t[0:i], y_train[0:i, 0], 'ro', label=r'data')
    plt2 = ax1.plot(t[0:i], B_ann[0:i, 0], 'g', label='random sample')

    if i == 0:
        ax1.legend(loc=1, fontsize=9)

    ims.append((plt1[0], plt2[0],))

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

###################
# Simulation flags
###################
train = True           #train the network
make_movie = True     #make a movie (of the training)
predict = False         #predict using the learned SGS term
store = False           #store the prediction results
make_movie_pred = False  #make a movie (of the prediction)

#####################
# Network parameters
#####################

#Feature engineering object - loads data file
feat_eng = es.methods.Feature_Engineering(load_data = True)
#get training data
h5f = feat_eng.get_hdf5_file()

#Large-scale and SGS data - convert to numpy array via [()]
X_data = h5f['X_data'][()]
B_data = h5f['B_data'][()]
n_steps = X_data.shape[0]

I = 6
X_data = X_data[:, I]
B_data = B_data[:, I]

lags = [[1, 10]]
max_lag = np.max(list(chain(*lags)))
X_lagged, y_train = feat_eng.lag_training_data([X_data], B_data, lags = lags)
Y_lagged, _ = feat_eng.lag_training_data([B_data], np.zeros(n_steps), 
                                         lags = lags, store = False)

ccm = es.methods.CCM(X_lagged, Y_lagged, [5, 5, 10, 10, 10], lags)
N_c = ccm.N_c
ccm.plot_2D_shadow_manifold()
ccm.compare_convex_hull_volumes()

"""
n_bins = 10
feat_eng.bin_data(y_train, n_bins)
sampler = es.methods.SimpleBin(feat_eng)

#number of softmax layers (one per output)
n_softmax = K

#number of output neurons 
n_out = n_bins*n_softmax

#train the neural network
if train:
    surrogate = es.methods.ANN(X=X_train, y=feat_eng.y_idx_binned, n_layers=4, n_neurons=256, 
                               n_softmax = K, n_out=K*n_bins, loss = 'cross_entropy',
                               activation='hard_tanh', batch_size=512,
                               lamb=0.0, decay_step=10**4, decay_rate=0.9, 
                               standardize_X=False, standardize_y=False, save=True)

    print('===============================')
    print('Training Quantized Softmax Network...')

    #train network for N_inter mini batches
    N_iter = 30000
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

    ims = []
    fig = plt.figure(figsize=[8,4])
    ax1 = fig.add_subplot(121, xlabel=r'bin number', ylabel=r'probability', 
                          ylim=[-0.05, 1.05])
    ax2 = fig.add_subplot(122, xlabel=r'time', ylabel=r'$B_k$')
    plt.tight_layout()

    #number of time steps to use in movie
    n_movie = 500

    #allocate memory
    samples = np.zeros([n_movie, n_softmax])

    #make movie by evaluating the network at TRAINING inputs
    for i in range(n_movie):
        
        #draw a random sample from the network - gives conditional
        #probability mass function (pmf)
        o_i, idx_max, idx = surrogate.get_softmax(X_train[i].reshape([1, n_feat]))
        idx_data = np.where(feat_eng.y_idx_binned[i] == 1.0)[0]
        
        #resample reference data based on conditional pmf
        samples[i, :] = sampler.resample(idx)

        if np.mod(i, 100) == 0:
            print('i =', i, 'of', n_movie)

        #create a single frame, store in 'ims'
        animate(i)

    #make a movie of all frame in 'ims'
    im_ani = animation.ArtistAnimation(fig, ims, interval=80, 
                                       repeat_delay=2000, blit=True)
    # im_ani.save('./movies/qsn.mp4')

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
    X_ann = np.zeros([t.size - max_lag, K])
    B_ann = np.zeros([t.size - max_lag, K])

    #start time integration
    idx = 0
    for t_i in t[max_lag:]:
 
        #get time lagged features from Feature Engineering object
        feat = feat_eng.get_feat_history()

        #SGS solve, draw random sample from network
        o_i, idx_max, rv_idx = surrogate.get_softmax(feat.reshape([1, n_feat]))
        B_n = sampler.resample(idx_max)
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
    
    #make a mavie of the coupled system    
    if make_movie_pred:
        
        ims = []
        fig = plt.figure(figsize=[4,4])
        ax1 = fig.add_subplot(111, xlabel=r'time', ylabel=r'$B_k$')
        plt.tight_layout()
        
        n_movie = 1000
        
        for i in range(n_movie):
            animate_pred(i)
            
        #make a movie of all frame in 'ims'
        im_ani = animation.ArtistAnimation(fig, ims, interval=80, 
                                           repeat_delay=2000, blit=True)
        # im_ani.save('./movies/qsn_pred.mp4')
"""
plt.show()