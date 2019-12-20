"""
===============================================================================
Applies the Quantized Softmax Network to the Kac Zwanzig heat bath model
===============================================================================
"""

def step(p, q, r):
    """
    
    Integrates v, p, u, q in time via sympletic Euler
    
    Parameters
    ----------
    p : (float): position distinguished particle.
    q : (float): momentum distinguished particle

    Returns
    -------
    
    p and q at next time level
    
    """    
    # potential V(q)=1/4 * (q^2-1)^2
    p = p - dt*q*(q**2 - 1) + dt*G**2*(r - N*q)
    q = q + dt*p
    
    return p, q

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

# Number of heat bath particles
N = 100

# Number of output data points
M = 10**5

# Coupling and mass scaling
G = 1.0

##################
# Time parameters
##################

dt = 0.01
t_end = M*dt
t = np.arange(0.0, t_end, dt)

#time lags per feature
lags = [list(np.arange(0, 100)), list(np.arange(0, 100))]
#lags = [[1], [1], [1]]
max_lag = np.max(list(chain(*lags)))

###################
# Simulation flags
###################
train = True           #train the network
make_movie = True      #make a movie (of the training)
predict = True        #predict using the learned SGS term
store = False           #store the prediction results
make_movie_pred = False  #make a movie (of the prediction)
online_training = True

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
X_train, y_train = feat_eng.lag_training_data([p_data, q_data], r_data, lags = lags)

X_mean = np.mean(X_train, axis = 0)
X_std = np.std(X_train, axis = 0)
# X_train, y_train = feat_eng.standardize_data(standardize_y = False)

n_bins = 10
feat_eng.bin_data(y_train, n_bins)
sampler = es.methods.SimpleBin(feat_eng)

#number of softmax layers (one per output)
n_softmax = 1

#number of output neurons 
n_out = n_bins*n_softmax

#train the neural network
if train:
    surrogate = es.methods.ANN(X=X_train, y=feat_eng.y_idx_binned, n_layers=2, 
                               n_neurons=256, 
                               n_softmax = n_softmax, n_out=n_softmax*n_bins, 
                               loss = 'cross_entropy',
                               activation='hard_tanh', batch_size=512,
                               lamb=0.0, decay_step=10**4, decay_rate=0.9, 
                               standardize_X=False, standardize_y=False, save=True)

    print('===============================')
    print('Training Quantized Softmax Network...')

    #train network for N_inter mini batches
    N_iter = 10000
    surrogate.train(N_iter, store_loss = True, sequential = True)

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
    print('Predicting with stochastic SGS model...')

    #features are time lagged, use the data to create initial feature set
    for i in range(max_lag):
        feat_eng.append_feat([[p_data[i]], [q_data[i]]], max_lag)

    #initial conditions
    q = q_data[max_lag]
    p = p_data[max_lag]

    #allocate memory for solutions
    q_ann = np.zeros(t.size - max_lag)
    p_ann = np.zeros(t.size - max_lag)
    r_ann = np.zeros(t.size - max_lag)

    if online_training:
        online_batch = 32
        surrogate.set_batch_size(online_batch)
        
        X_online = np.zeros([online_batch, surrogate.n_in])
        y_online = np.zeros([online_batch, surrogate.n_out])
        
        J = 0        
    
    #start time integration
    idx = 0
    for t_i in t[max_lag:]:
 
        #get time lagged features from Feature Engineering object
        feat = feat_eng.get_feat_history()
        
        if online_batch:
            X_online[J, :] = feat
            y_online[J, :] = surrogate.y[idx, :]
            
            J += 1
            
            if J == online_batch:
                print('Online back prop')
                J = 0                
                surrogate.batch(X_online, y_online.T)

        # feat = (feat - X_mean)/X_std

        #SGS solve, draw random sample from network
        o_i, idx_max, rv_idx = surrogate.get_softmax(feat.reshape([1, n_feat]))
        r = sampler.resample(rv_idx)[0]
        
        #solve large-scale equation
        p, q = step(p, q, r)
        
        #append the features to the Feature Engineering object
        feat_eng.append_feat([[p], [q]], max_lag)

        #store solutions
        q_ann[idx] = q
        p_ann[idx] = p
        r_ann[idx] = r

        idx += 1

        if np.mod(idx, 1000) == 0:
            print('t =', np.around(t_i, 1), 'of', t_end)

    print('done')

    #############   
    # Plot PDEs #
    #############

    print('===============================')
    print('Postprocessing results')

    fig = plt.figure(figsize=[12, 4])
    ax1 = fig.add_subplot(131, xlabel=r'p')
    ax2 = fig.add_subplot(132, xlabel=r'q')
    ax3 = fig.add_subplot(133, xlabel=r'r')

    post_proc = es.methods.Post_Processing()

    p_dom_surr, p_pde_surr = post_proc.get_pde(p_ann.flatten()[0:-1:10])
    p_dom, p_pde = post_proc.get_pde(p_data.flatten()[0:-1:10])

    q_dom_surr, q_pde_surr = post_proc.get_pde(q_ann.flatten()[0:-1:10])
    q_dom, q_pde = post_proc.get_pde(q_data.flatten()[0:-1:10])

    r_dom_surr, r_pde_surr = post_proc.get_pde(r_ann.flatten()[0:-1:10])
    r_dom, r_pde = post_proc.get_pde(r_data.flatten()[0:-1:10])

    ax1.plot(p_dom, p_pde, 'ko', label='Kac Zwanzig')
    ax1.plot(p_dom_surr, p_pde_surr, label='ANN')
    ax2.plot(q_dom, q_pde, 'ko', label='Kac Zwanzig')
    ax2.plot(q_dom_surr, q_pde_surr, label='ANN')
    ax3.plot(r_dom, r_pde, 'ko', label='Kac Zwanzig')
    ax3.plot(r_dom_surr, r_pde_surr, label='ANN')

    plt.yticks([])

    plt.legend(loc=0)

    plt.tight_layout()

    #############   
    # Plot ACFs #
    #############

    fig = plt.figure()
    ax = fig.add_subplot(111, ylabel='ACF', xlabel='time')

    R_p_data = post_proc.auto_correlation_function(p_data, max_lag=1000)
    R_ann = post_proc.auto_correlation_function(p_ann, max_lag=1000)

    dom_acf = np.arange(R_p_data.size)*dt

    ax.plot(dom_acf, R_p_data, 'ko', label='Kac Zwanzig')
    ax.plot(dom_acf, R_ann, label='ANN')

    leg = plt.legend(loc=0)

    plt.tight_layout()

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