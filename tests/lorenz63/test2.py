
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from itertools import chain
import matplotlib.pyplot as plt
import easysurrogate as es


def one_hot(idx):

    # this should be the non-empty binnumbers
    unique_idx = np.unique(idx)
    B = unique_idx.size
    S = idx.size

    y_idx_binned = np.zeros([S, B])

    count = 0

    for idx_i in unique_idx:
        i = np.where(idx == idx_i)[0]
        y_idx_binned[i, count] = 1.0
        count += 1

    return y_idx_binned


plt.close('all')

#####################
# Network parameters
#####################

# Feature engineering object - loads data file
feat_eng = es.methods.Feature_Engineering(load_data=True)
# get training data
h5f = feat_eng.get_hdf5_file()

# Large-scale and SGS data - convert to numpy array via [()]
X_data = h5f['X_data'][()]
# Y_data = h5f['Y_data'][()]
B_data = h5f['B_data'][()]
n_steps = X_data.shape[0]

I = 6

lags = [range(1, 75)]
lags_y = [[1, 10]]
max_lag = np.max(list(chain(*lags)))
X_lagged, y_train = feat_eng.lag_training_data([X_data[:, I]], B_data[:, I], lags=lags)
Y_lagged, _ = feat_eng.lag_training_data([B_data[:, I]], np.zeros(n_steps),
                                         lags=lags_y, store=False)

ccm = es.methods.CCM(Y_lagged, np.zeros(n_steps), [10, 10], lags_y)
# N_c = ccm.N_c
ccm.plot_2D_binning_object()
# ccm.plot_2D_shadow_manifold()
# ccm.compare_convex_hull_volumes()

one_hot_enc = one_hot(ccm.binnumber)

n_out = one_hot_enc.shape[1]

surrogate = es.methods.ANN(X=X_lagged, y=one_hot_enc, n_layers=4, n_neurons=256,
                           n_softmax=1, n_out=n_out, loss='cross_entropy',
                           activation='leaky_relu', batch_size=512,
                           lamb=0.0, decay_step=10**4, decay_rate=0.9,
                           standardize_X=True, standardize_y=False, save=False)

print('===============================')
print('Training Quantized Softmax Network...')

# train network for N_inter mini batches
N_iter = 5000
surrogate.train(N_iter, store_loss=True)

n_train = X_lagged.shape[0]
n_feat = X_lagged.shape[1]

pred = np.zeros(n_train)

for i in range(n_train):
    o_i, idx_max, rvs = surrogate.get_softmax(surrogate.X[i].reshape([1, n_feat]))

    pred[i] = idx_max[0]

plt.figure()

for i in np.unique(pred):

    idx = np.where(pred == i)[0]

    plt.plot(X_lagged[idx, 0], X_lagged[idx, 0], '+')

"""

# feat_eng.estimate_embedding_dimension(X[0:-1:10, 0], 6)

#################################
# Run full model to generate IC #
#################################

X_n = np.zeros(3)
#initial condition of the training data
# X_n[0] = 0.0; X_n[1]  = 1.0; X_n[2] = 1.05

#new initial condition to break symmetry
X_n[0] = 0.20; X_n[1] = 0.75; X_n[2] = 1.0

#initial condition right-hand side
f_nm1 = rhs(X_n)

for n in range(max_lag):

    #step in time
    X_np1, f_n = step(X_n, f_nm1)

    feat_eng.append_feat([[X_np1[0]]], max_lag)

    #update variables
    X_n = X_np1
    f_nm1 = f_n

########################################
# Run the model with the CCM surrogate #
########################################

#number of time steps
n_pred = n_steps - max_lag

#reduce IC to X only
X_n = X_n[0]
f_nm1 = f_nm1[0]

X_surr = np.zeros([n_pred, 1])

for i in range(n_pred):

    #step in time
    X_np1, f_n = step_ccm(X_n, f_nm1)

    #update variables
    X_n = X_np1
    f_nm1 = f_n

    X_surr[i, :] = X_n

#############
# Plot PDEs #
#############

post_proc = es.methods.Post_Processing()

print('===============================')
print('Postprocessing results')

fig = plt.figure(figsize=[4, 4])

ax = fig.add_subplot(111, xlabel=r'$x$')
X_dom_surr, X_pde_surr = post_proc.get_pde(X_surr[0:-1:10, 0])
X_dom, X_pde = post_proc.get_pde(X[0:-1:10, 0])
ax.plot(X_dom, X_pde, 'ko', label='L63')
ax.plot(X_dom_surr, X_pde_surr, label='ANN')
plt.yticks([])
plt.legend(loc=0)

plt.tight_layout()

#############
# Plot ACFs #
#############

fig = plt.figure(figsize=[4, 4])

ax = fig.add_subplot(111, ylabel='ACF X', xlabel='time')
R_data = post_proc.auto_correlation_function(X[:, 0], max_lag = 500)
R_sol = post_proc.auto_correlation_function(X_surr[:, 0], max_lag = 500)
dom_acf = np.arange(R_data.size)*dt
ax.plot(dom_acf, R_data, 'ko', label='L63')
ax.plot(dom_acf, R_sol, label='ANN')
leg = plt.legend(loc=0)

plt.tight_layout()
"""
plt.show()
