"""
===============================================================================
Applies the Resampler surrogate to the Kac Zwanzig heat bath model
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
    
    plt1 = ax1.plot(t[0:i], y_train[0:i], 'ro', label=r'data')
    plt2 = ax1.plot(t[0:i], samples[0:i], 'g', label='random sample')

    if i == 0:
        ax1.legend(loc=1, fontsize=9)

    ims.append((plt1, plt2[0],))
    
###############
# Main program
###############
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import easysurrogate as es
import h5py, os
from itertools import chain, product
from scipy.stats import norm, rv_discrete, binned_statistic

plt.close('all')

########################
# Kac Zwanzig parameters
########################

# Number of heat bath particles
N = 100

# Number of output data points
M = 10**6

# Coupling and mass scaling
G = 1.0

##################
# Time parameters
##################

dt = 0.01
t_end = M*dt
t = np.arange(0.0, t_end, dt)

#time lags per feature
lags = [[1], [1, 2]]
max_lag = np.max(list(chain(*lags)))

###################
# Simulation flags
###################
train = True            #train the network
make_movie = False       #make a movie (of the training)
predict = True          #predict using the learned SGS term
store = False           #store the prediction results

#####################
# Network parameters
#####################

#Feature engineering object - loads data file
feat_eng = es.methods.Feature_Engineering()
#get training data
h5f = feat_eng.get_hdf5_file()

#Large-scale and SGS data - convert to numpy array via [()]
q_data = h5f['q_n'][()]
p_data = h5f['p_n'][()]
r_data = h5f['r_n'][()]

#Lag features as defined in 'lags'
X_train, y_train = feat_eng.lag_training_data([q_data, r_data], r_data, lags = lags)
X_mean, X_std = feat_eng.moments_lagged_features([q_data, r_data], lags = lags)

#number of input features
n_feat = X_train.shape[1]

#number of bins
n_bins = 10

surrogate = es.methods.Resampler(X_train, y_train.flatten(), 1, n_bins, lags)
surrogate.print_bin_info()
surrogate.plot_2D_binning_object()

#Post processing object
post_proc = es.methods.Post_Processing()

#if True make a movie of the solution
if make_movie:
    
    print('===============================')
    print('Making movie...')

    ims = []
    fig = plt.figure()
    ax1 = fig.add_subplot(111, xlabel=r'time', ylabel=r'$B_k$')
    plt.tight_layout()

    #number of time steps to use in movie
    n_movie = 500

    #allocate memory
    samples = np.zeros(n_movie)

    #make movie by evaluating the network at TRAINING inputs
    for i in range(n_movie):
        
        samples[i] = surrogate.get_sample(X_train[i].reshape([1, n_feat]))[0]
        
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
        feat_eng.append_feat([[q_data[i]], [r_data[i]]], max_lag)

    #initial conditions
    q = q_data[max_lag]
    p = p_data[max_lag]

    #allocate memory for solutions
    q_surr = np.zeros(t.size - max_lag)
    p_surr = np.zeros(t.size - max_lag)
    r_surr = np.zeros(t.size - max_lag)
   
    #start time integration
    idx = 0
    for t_i in t[max_lag:]:
 
        #get time lagged features from Feature Engineering object
        feat = feat_eng.get_feat_history()
        feat = np.array(list(chain(*feat)))
        
        r = surrogate.get_sample(feat.reshape([1, n_feat]))[0]

        #solve large-scale equation
        p, q = step(p, q, r)
        
        #append the features to the Feature Engineering object
        feat_eng.append_feat([[q], [r]], max_lag)

        #store solutions
        q_surr[idx] = q
        p_surr[idx] = p
        r_surr[idx] = r

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

    p_dom_surr, p_pde_surr = post_proc.get_pde(p_surr.flatten())
    p_dom, p_pde = post_proc.get_pde(p_data.flatten())

    q_dom_surr, q_pde_surr = post_proc.get_pde(q_surr.flatten())
    q_dom, q_pde = post_proc.get_pde(q_data.flatten())

    r_dom_surr, r_pde_surr = post_proc.get_pde(r_surr.flatten())
    r_dom, r_pde = post_proc.get_pde(r_data.flatten())

    ax1.plot(p_dom, p_pde, 'ko', label='Kac Zwanzig')
    ax1.plot(p_dom_surr, p_pde_surr, label='Surr')
    ax2.plot(q_dom, q_pde, 'ko', label='Kac Zwanzig')
    ax2.plot(q_dom_surr, q_pde_surr, label='Surr')
    ax3.plot(r_dom, r_pde, 'ko', label='Kac Zwanzig')
    ax3.plot(r_dom_surr, r_pde_surr, label='Surr')

    plt.yticks([])

    plt.legend(loc=0)

    plt.tight_layout()

    #############   
    # Plot ACFs #
    #############

    fig = plt.figure()
    ax = fig.add_subplot(111, ylabel='ACF', xlabel='time')

    R_p_data = post_proc.auto_correlation_function(p_data, max_lag=1000)
    R_ann = post_proc.auto_correlation_function(p_surr, max_lag=1000)

    dom_acf = np.arange(R_p_data.size)*dt

    ax.plot(dom_acf, R_p_data, 'ko', label='Kac Zwanzig')
    ax.plot(dom_acf, R_ann, label='ANN')

    leg = plt.legend(loc=0)

    plt.tight_layout()

plt.show()