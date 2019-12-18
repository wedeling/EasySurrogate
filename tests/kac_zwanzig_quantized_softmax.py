"""
===============================================================================
Applies the Quantized Softmax Network to the Kac Zwanzig heat bath model
===============================================================================
"""

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
    plt2 = ax1.plot(idx_data[0], 0.0, 'ro', label=r'data')
    plt3 = ax2.plot(t[0:i], y_train[0:i], 'ro', label=r'data')
    plt4 = ax2.plot(t[0:i], samples[0:i], 'g', label='random sample')

    if i == 0:
        ax1.legend(loc=1, fontsize=9)
        ax2.legend(loc=1, fontsize=9)

    ims.append((plt1, plt2[0], plt3[0], plt4[0],))

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

########################
# Kac Zwanzig parameters
########################

# Number of output data points
M = 10**5

##################
# Time parameters
##################

dt = 0.01
t_end = M*dt
t = np.arange(0.0, t_end, dt)

#time lags per feature
lags = [list(np.arange(100)), list(np.arange(100))]
max_lag = np.max(list(chain(*lags)))

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
feat_eng = es.methods.Feature_Engineering()
#get training data
h5f = feat_eng.get_hdf5_file()

#Large-scale and SGS data - convert to numpy array via [()]
q_data = h5f['q'][()]
p_data = h5f['p'][()]
r_data = h5f['r'][()]

#Lag features as defined in 'lags'
X_train, y_train = feat_eng.lag_training_data([q_data, p_data], r_data, lags = lags)
X_train, y_train = feat_eng.standardize_data(standardize_y = False)

n_bins = 10
feat_eng.bin_data(y_train, n_bins)
sampler = es.methods.SimpleBin(feat_eng)

#number of softmax layers (one per output)
n_softmax = 1

#number of output neurons 
n_out = n_bins*n_softmax

#train the neural network
if train:
    surrogate = es.methods.ANN(X=X_train, y=feat_eng.y_idx_binned, n_layers=6, 
                               n_neurons=50, 
                               n_softmax = n_softmax, n_out=n_softmax*n_bins, 
                               loss = 'cross_entropy',
                               activation='hard_tanh', batch_size=512,
                               lamb=0.0, decay_step=10**4, decay_rate=0.9, 
                               standardize_X=False, standardize_y=False, save=True)

    print('===============================')
    print('Training Quantized Softmax Network...')

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
    # im_ani.save('./movies/qsn_kac.mp4')

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

plt.show()