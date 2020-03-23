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
        
    rhs_X += h_x*B
        
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

# K = 32
# J = 16
# F = 18.0
# h_x = -3.2
# h_y = 1.0
# epsilon = 0.5

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
train = True           #train the network
make_movie = False      #make a movie (of the training)
predict = True          #predict using the learned SGS term
store = True            #store the prediction 
make_movie_pred = False #make a movie (of the prediction)

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

#Lag features as defined in 'lags'
X_train, y_train = feat_eng.lag_training_data([X_data], B_data, lags = lags)

#number of bins per B_k
n_bins = 10
#one-hot encoded training data per B_k
feat_eng.bin_data(y_train, n_bins)
#simple sampler to draw random samples from the bins
sampler = es.methods.SimpleBin(feat_eng)

#number of softmax layers (one per output)
n_softmax = K

#number of output neurons 
n_out = n_bins*n_softmax

#test set fraction
test_frac = 0.5
n_train = np.int(X_train.shape[0]*(1.0 - test_frac))

#train the neural network
if train:
    surrogate = es.methods.ANN(X=X_train[0:n_train], y=feat_eng.y_idx_binned[0:n_train],
                               n_layers=4, n_neurons=256, 
                               n_softmax = K, n_out=K*n_bins, loss = 'cross_entropy',
                               activation='leaky_relu', batch_size=512,
                               lamb=0.0, decay_step=10**4, decay_rate=0.9, 
                               standardize_X=True, standardize_y=False, save=True)

    print('===============================')
    print('Training Quantized Softmax Network...')

    #train network for N_inter mini batches
    N_iter = 10000
    surrogate.train(N_iter, store_loss = True)
    surrogate.compute_misclass_softmax()

#load a neural network from disk
else:
    #first create dummy ANN object    
    surrogate = es.methods.ANN(X=X_train, y=y_train)
    #the load trained network from disk
    surrogate.load_ANN()
    
X_mean = surrogate.X_mean
X_std = surrogate.X_std
    
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
        feat = feat_eng.get_feat_history(max_lag)
        
        feat = (feat - X_mean)/X_std

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
    post_proc = es.methods.Post_Processing(load_data = False)

else:
    post_proc = es.methods.Post_Processing(load_data = True)
    h5f = post_proc.get_hdf5_file()
    X_ann = h5f['X'][()]
    B_ann = h5f['B'][()]

#############   
# Plot PDEs #
#############

print('===============================')
print('Postprocessing results')

#starting index of the ACF / CCF calculations
start_idx = n_train     #set to the end of the training / beginning test set

fig = plt.figure(figsize=[8,4])
ax = fig.add_subplot(121, xlabel=r'$X_k$')
X_dom_surr, X_pde_surr = post_proc.get_pdf(X_ann[start_idx:-1:10].flatten())
X_dom, X_pde = post_proc.get_pdf(X_data[start_idx:-1:10].flatten())
ax.plot(X_dom, X_pde, 'ko', label='L96')
ax.plot(X_dom_surr, X_pde_surr, label='QSN')
plt.yticks([])
plt.legend(loc=0)

ax = fig.add_subplot(122, xlabel=r'$r_k$')
B_dom_surr, B_pde_surr = post_proc.get_pdf(B_ann[start_idx:-1:10].flatten())
B_dom, B_pde = post_proc.get_pdf(B_data[start_idx:-1:10].flatten())
ax.plot(B_dom, B_pde, 'ko', label='L96')
ax.plot(B_dom_surr, B_pde_surr, label='QSN')
plt.yticks([])
plt.legend(loc=0)

plt.tight_layout()

#############   
# Plot ACFs #
#############

acf_lag = 1000

fig = plt.figure(figsize=[8,4])
ax1 = fig.add_subplot(121, ylabel='$\mathrm{ACF}\;X_k$', xlabel='time')
ax2 = fig.add_subplot(122, ylabel='$\mathrm{ACF}\;r_k$', xlabel='time')

acf_X_data = np.zeros(acf_lag-1); acf_X_sol = np.zeros(acf_lag-1)
acf_B_data = np.zeros(acf_lag-1); acf_B_sol = np.zeros(acf_lag-1)

#average over all spatial points
for k in range(K):
    print('k=%d'%k)
    acf_X_data += 1/K*post_proc.auto_correlation_function(X_data[start_idx:, k], max_lag=acf_lag)
    acf_X_sol += 1/K*post_proc.auto_correlation_function(X_ann[start_idx:, k], max_lag=acf_lag)
    
    acf_B_data += 1/K*post_proc.auto_correlation_function(B_data[start_idx:, k], max_lag=acf_lag)
    acf_B_sol += 1/K*post_proc.auto_correlation_function(B_ann[start_idx:, k], max_lag=acf_lag)

dom_acf = np.arange(acf_lag-1)*dt
ax1.plot(dom_acf, acf_X_data, 'ko', label='L96')
ax1.plot(dom_acf, acf_X_sol, label='QSN')
leg = plt.legend(loc=0)

ax2.plot(dom_acf, acf_B_data, 'ko', label='L96')
ax2.plot(dom_acf, acf_B_sol, label='QSN')
leg = plt.legend(loc=0)

plt.tight_layout()

#############   
# Plot CCFs #
#############

fig = plt.figure(figsize=[8,4])
ax1 = fig.add_subplot(121, ylabel='$\mathrm{CCF}\;X_k$', xlabel='time')
ax2 = fig.add_subplot(122, ylabel='$\mathrm{CCF}\;r_k$', xlabel='time')

ccf_X_data = np.zeros(acf_lag-1); ccf_X_sol = np.zeros(acf_lag-1)
ccf_B_data = np.zeros(acf_lag-1); ccf_B_sol = np.zeros(acf_lag-1)

#average over all spatial points
for k in range(K-1):
    print('k=%d'%k)

    ccf_X_data += 1/K*post_proc.cross_correlation_function(X_data[start_idx:, k], X_data[start_idx:,k+1], max_lag=1000)
    ccf_X_sol += 1/K*post_proc.cross_correlation_function(X_ann[start_idx:, k], X_ann[start_idx:, k+1], max_lag=1000)
    
    ccf_B_data += 1/K*post_proc.cross_correlation_function(B_data[start_idx:, k], B_data[start_idx:,k+1], max_lag=1000)
    ccf_B_sol += 1/K*post_proc.cross_correlation_function(B_ann[start_idx:, k], B_ann[start_idx:, k+1], max_lag=1000)

dom_ccf = np.arange(acf_lag - 1)*dt
ax1.plot(dom_ccf, ccf_X_data, 'ko', label='L96')
ax1.plot(dom_ccf, ccf_X_sol, label='QSN')

ax2.plot(dom_ccf, ccf_B_data, 'ko', label='L96')
ax2.plot(dom_ccf, ccf_B_sol, label='QSN')
leg = plt.legend(loc=0)

plt.tight_layout()

############################
# store simulation results #
############################
if store:
    samples = {'X': X_ann, 'B':B_ann, \
               'dom_acf':dom_acf, 'acf_X_data':acf_X_data, 'acf_X_ann':acf_X_sol, 
               'dom_ccf':dom_ccf, 'ccf_X_data':ccf_X_data, 'ccf_X_ann':ccf_X_sol, 
               'acf_B_data':acf_B_data, 'acf_B_ann':acf_B_sol, 
               'ccf_B_data':ccf_B_data, 'ccf_B_ann':ccf_B_sol, 
               }
    post_proc.store_samples_hdf5(samples)

#make a mavie of the coupled system    
if make_movie_pred:
    
    ims = []
    fig = plt.figure(figsize=[4, 4])
    ax1 = fig.add_subplot(111, xlabel=r'time', ylabel=r'$B_k$')
    plt.tight_layout()
    
    n_movie = 1000
    
    for i in range(n_movie):
        animate_pred(i)
        
    #make a movie of all frame in 'ims'
    im_ani = animation.ArtistAnimation(fig, ims, interval=80, 
                                       repeat_delay=2000, blit=True)
    im_ani.save('../movies/qsn_pred.mp4')

plt.show()