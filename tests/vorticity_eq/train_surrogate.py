import numpy as np
import easysurrogate as es
import h5py, os
from itertools import chain, product
from scipy.stats import norm, rv_discrete
import matplotlib.pyplot as plt
from matplotlib import animation

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
    plt1 = ax1.vlines(np.arange(max(np.unique(feat_eng.binnumbers[:, 0]))), \
    	ymin = np.zeros(max(np.unique(feat_eng.binnumbers[:, 0]))), ymax = o_i[0],
                      colors = c, label=r'conditional pmf')
    plt2 = ax1.plot(idx_data[0], 0.0, 'ro', label=r'SGS data')
    plt3 = ax2.plot(t[0:i], abs(y_train[0:i]), 'ro', alpha = 0.5, label=r'SGS data')
    plt4 = ax2.plot(t[0:i], abs(samples[0:i, 0]), 'b', label='random sample')

    if i == 0:
        ax1.legend(loc=1, fontsize=9)
        ax2.legend(loc=1, fontsize=9)

    ims.append((plt1, plt2[0], plt4[0], plt3[0],))

####################
# Simulation flags #
####################
train = True             #train the network
make_movie = True        #make a movie (of the training)
predict = False          #predict using the learned SGS term
store = False            #store the prediction 
make_movie_pred = False  #make a movie (of the prediction)

######################
# Network parameters #
######################

#Feature engineering object - loads data file
feat_eng = es.methods.Feature_Engineering(load_data = True)

#get training data
h5f = feat_eng.get_hdf5_file()

#filter out all the zeros from the 3/2 rule
I, J = np.where(h5f['P_LF'][()] == 1.0)

#extract features
vort = h5f['w_hat_nm1_LF'][()][:, I, J]
jac = h5f['VgradW_hat_nm1_LF'][()][:, I, J]
sgs = h5f['r_hat_nm1'][()][:, I, J]

# Reshape features into scalars
vort = vort.flatten()
jac = jac.flatten()
sgs = sgs.flatten()

##################
# Time parameters
##################
Omega = 7.292*10**-5 #time scale
day = 24*60**2*Omega

dt = 0.25*day 
t0 = 365*day
t_end = 4*365*day
t = np.arange(t0, t_end, dt)

#Lag features as defined in 'lags'
lags = [[1,10], [1,10]]
max_lag = np.max(list(chain(*lags)))

# X_train, y_train = feat_eng.lag_training_data([jac], sgs, lags = lags)
jac_re = jac.real.flatten()
jac_im = jac.imag.flatten()
X_train, y_train = feat_eng.lag_training_data([jac_re, jac_im], sgs, lags = lags)

#number of bins per B_k
n_bins = 5

#one-hot encoded training data per B_k
feat_eng.bin_data(y_train, n_bins)
#simple sampler to draw random samples from the bins
sampler = es.methods.SimpleBin(feat_eng)


#number of softmax layers (one per output)
n_softmax = 1

#number of output neurons 
n_out = feat_eng.y_idx_binned.shape[1]

#test set fraction
test_frac = 0.5
n_train = np.int(X_train.shape[0]*(1.0 - test_frac))

print(feat_eng.y_idx_binned[10]) 
#train the neural network
if train:
    surrogate = es.methods.ANN(X=X_train[0:n_train], y=feat_eng.y_idx_binned[0:n_train],
                               n_layers=2, n_neurons=20, 
                               n_softmax=n_softmax, n_out=n_out, loss='cross_entropy',
                               activation='leaky_relu', batch_size=512,
                               lamb=0.0, decay_step=10**4, decay_rate=0.9, 
                               standardize_X=True, standardize_y=False, save=False)

    print('=====================================')
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

"""
X_mean = surrogate.X_mean
X_std = surrogate.X_std
    
#number of features
n_feat = surrogate.n_in

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
    samples = np.zeros([n_movie, n_softmax],dtype=y_train.dtype)

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
    plt.show()
    # im_ani.save('./movies/test.mp4')

    print('done')
"""
