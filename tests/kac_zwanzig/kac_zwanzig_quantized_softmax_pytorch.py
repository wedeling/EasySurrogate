import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    
    def __init__(self, n_inputs, n_neurons):
        super(Net, self).__init__()
        
        self.n_inputs = n_inputs
        
        self.l1 = nn.Linear(n_inputs, n_neurons)
        self.l2 = nn.Linear(n_neurons, n_neurons)
        self.l3 = nn.Linear(n_neurons, n_neurons)
        self.l4 = nn.Linear(n_neurons, n_bins)
        
    def forward(self, x):
        
        x = x.view(-1, self.n_inputs)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.log_softmax(self.l4(x))
        
        return x        

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
    
    plt1 = ax1.vlines(np.arange(n_bins), ymin = np.zeros(n_bins), ymax = np.exp(o_i.detach().numpy()[0]),
                      colors = c, label=r'conditional pmf')
    plt2 = ax1.plot(idx_data[0], 0.0, 'ro', label=r'data')
    plt3 = ax2.plot(t[0:i], y_train[0:i], 'ro', label=r'data')
    plt4 = ax2.plot(t[0:i], samples[0:i], 'g', label='random sample')

    if i == 0:
        ax1.legend(loc=1, fontsize=9)
        ax2.legend(loc=1, fontsize=9)

    ims.append((plt1, plt2[0], plt3[0], plt4[0],))
    
def animate_pred(i):
    """
    Generate a movie frame for the coupled system
    """
    plt1 = ax1.plot(t[0:i], y_train[0:i], 'ro', label=r'data')
    plt2 = ax1.plot(t[0:i], r_ann[0:i], 'g', label='random sample')

    if i == 0:
        ax1.legend(loc=1, fontsize=9)

    ims.append((plt1[0], plt2[0],))
    
def get_feat_mids(feats, mids):
    
    mid_feats = torch.zeros(feats.shape)
    
    idx = 0
    for feat in feats:
        
        j = torch.abs(feat - mids[idx]).argmin()
        mid_feats[idx] = mids[idx][j]
        idx += 1
    
    return mid_feats
  
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

dt = 0.0001
t_end = M*dt
t = np.arange(0.0, t_end, dt)

#time lags per feature
lags = [range(1, 10), range(1, 10)]
max_lag = np.max(list(chain(*lags)))

###################
# Simulation flags
###################
train = True        #train the network
make_movie = True      #make a movie (of the training)
predict = True        #predict using the learned SGS term
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
q_data = h5f['q_n'][()]
p_data = h5f['p_n'][()]
r_data = h5f['r_n'][()]

# n_bins_p = 10
# p_bins = np.linspace(np.min(p_data), np.max(p_data), n_bins_p+1)
# _, _, p_data_binned = binned_statistic(p_data, np.zeros(p_data.size), bins=p_bins)
# p_mids = 0.5*(p_bins[1:] + p_bins[0:-1])
# p_mids_data = p_mids[p_data_binned-1]

# n_bins_q = 10
# q_bins = np.linspace(np.min(q_data), np.max(q_data), n_bins_q+1)
# _, _, q_data_binned = binned_statistic(q_data, np.zeros(q_data.size), bins=q_bins)
# q_mids = 0.5*(q_bins[1:] + q_bins[0:-1])
# q_mids_data = q_mids[q_data_binned-1]

# n_bins_r = 10
# r_bins = np.linspace(np.min(r_data), np.max(r_data), n_bins_r+1)
# _, _, r_data_binned = binned_statistic(r_data, np.zeros(r_data.size), bins=r_bins)
# r_mids = 0.5*(r_bins[1:] + r_bins[0:-1])
# r_mids_data = r_mids[r_data_binned-1]

#Lag features as defined in 'lags'
X_train, y_train = feat_eng.lag_training_data([q_data, r_data], r_data, lags = lags)
# X_mean, X_std = feat_eng.moments_lagged_features([q_data, r_data], lags = lags)

# X_train = (X_train - X_mean)/X_std

# X_mean = np.mean(X_train, axis = 0)
# X_std = np.std(X_train, axis = 0)
# X_train, _ = feat_eng.standardize_data(standardize_y = False)

n_bins = 10
feat_eng.bin_data(y_train, n_bins)
sampler = es.methods.SimpleBin(feat_eng)

#number of softmax layers (one per output)
n_softmax = 1

#number of output neurons 
n_out = n_bins*n_softmax

###########
# Pytorch #
###########

#Must store as float (single precision?), or use net.double()
X_train = torch.from_numpy(X_train).float()

mids = []
for i in range(X_train.size(1)):
    mids.append(torch.unique(X_train[:,i]))

#Must store as long
y_train_bins = torch.from_numpy(feat_eng.binnumbers - 1).long()

#create a neural net
n_inputs = X_train.size(1)
net = Net(n_inputs = n_inputs, n_neurons = 100)

#cross entropy loss function
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

n_train = X_train.size()[0]

batch_size = 64

def train(epoch):
    net.train()
    for i in range(int(n_train/batch_size)):
        # data, target = data.to(device), target.to(device)
        
        idx = np.random.randint(0, n_train, batch_size)
        
        feats = X_train[idx]
        target = y_train_bins[idx]
        target = target.view(batch_size)
        
        optimizer.zero_grad()
        output = net(feats)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if np.mod(i, 100) == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, i , n_train/batch_size,
                100. * i / n_train/batch_size, loss.item()))
            
def test():
    net.eval()
    test_loss = 0
    correct = 0
    for i in range(n_train):
        # data, target = data.to(device), target.to(device)
        feats = X_train[i]
        target = y_train_bins[i]
        
        output = net(feats)
        
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= n_train
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{n_train} '
          f'({100. * correct / n_train:.0f}%)')
    
n_epoch = 1

for i in range(n_epoch):
    train(i)
    #test()

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
        o_i = net.forward(X_train[i])
        # o_i = np.exp(o_i.detach().numpy())
        idx_max = [torch.argmax(o_i).item()]
        
        idx_data = feat_eng.binnumbers[i] - 1
        
        #resample reference data based on conditional pmf
        samples[i, :] = sampler.resample(idx_max)

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
    q_ann = np.zeros(t.size - max_lag)
    p_ann = np.zeros(t.size - max_lag)
    r_ann = np.zeros(t.size - max_lag)
   
    #start time integration
    idx = 0
    for t_i in t[max_lag:]:
 
        #get time lagged features from Feature Engineering object
        feat = feat_eng.get_feat_history()
        # feat = torch.from_numpy(feat).float()
        # feat_mids = get_feat_mids(feat, mids)
        
        # feat = (feat - X_mean)/X_std

        #SGS solve, draw random sample from network
        o_i = net.forward(feat)
        # o_i = o_i.detach().numpy()
        # o_i = np.exp(o_i)/np.sum(np.exp(o_i))        
        idx_max = torch.argmax(o_i).item()
        
        print(idx_max)
        
        r = sampler.resample([idx_max])[0]
        # r = r_data[idx+max_lag]
        
        #solve large-scale equation
        p, q = step(p, q, r)
        
        #append the features to the Feature Engineering object
        feat_eng.append_feat([[q], [r]], max_lag)

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

    p_dom_surr, p_pde_surr = post_proc.get_pde(p_ann.flatten())
    p_dom, p_pde = post_proc.get_pde(p_data.flatten())

    q_dom_surr, q_pde_surr = post_proc.get_pde(q_ann.flatten())
    q_dom, q_pde = post_proc.get_pde(q_data.flatten())

    r_dom_surr, r_pde_surr = post_proc.get_pde(r_ann.flatten())
    r_dom, r_pde = post_proc.get_pde(r_data.flatten())

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