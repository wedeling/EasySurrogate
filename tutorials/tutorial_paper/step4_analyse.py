
def make_movie(qsn_surrogate, n_frames=500):
    """
    Makes a move using the training data. Left subplot shows the evolution of the
    kernel-density estimate, and the right subplot show the time series of the data
    and the random samples drawm from the kernel-density estimate. Saves the movie
    to a .gif file.

    Parameters

    n_frames (int): default is 500
        The number of frames to use in the movie.

    Returns: None
    """

    # get the (normalized, time-lagged) training data from the neural network
    X = qsn_surrogate.neural_net.X
    y = qsn_surrogate.neural_net.y

    print('===============================')
    print('Making movie...')

    # list to store the movie frames in
    ims = []
    fig = plt.figure(figsize=[8, 4])
    ax1 = fig.add_subplot(121, xlabel=r'$B_k$', ylabel=r'', yticks=[])
    ax2 = fig.add_subplot(122, xlabel=r'time', ylabel=r'$B_k$')
    plt.tight_layout()

    # number of features
    n_feat = X.shape[1]
    # number of softmax layers
    n_softmax = qsn_surrogate.n_softmax

    # allocate memory
    samples = np.zeros([n_frames])

    # make movie by evaluating the network at TRAINING inputs
    for i in range(n_frames):

        # draw a random sample from the network
        o_i, idx_max, _ = qsn_surrogate.neural_net.get_softmax(X[i].reshape([1, n_feat]))
        samples[i] = qsn_surrogate.sampler.resample(idx_max)[0]
        if np.mod(i, 100) == 0:
            print('i =', i, 'of', n_frames)

        # create a single frame, store in 'ims'
        plt2 = ax1.plot(range(qsn_surrogate.n_bins), 
                        np.zeros(qsn_surrogate.n_bins),
                        o_i[0], 'b', label=r'conditional pmf')
        plt3 = ax1.plot(y[i][0], 0.0, 'ro', label=r'data')
        plt4 = ax2.plot(y[0:i, 0], 'ro', label=r'data')
        plt5 = ax2.plot(samples[0:i, 0], 'g', label='random sample')

        if i == 0:
            ax1.legend(loc=1, fontsize=9)
            ax2.legend(loc=1, fontsize=9)

        ims.append((plt2[0], plt3[0], plt4[0], plt5[0],))

    # make a movie of all frame in 'ims'
    im_ani = animation.ArtistAnimation(fig, ims, interval=20,
                                       repeat_delay=2000, blit=True)
    im_ani.save('kvm.gif')
    print('done. Saved movie to qsn.gif')

import os
import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es
from matplotlib import animation

home = os.path.abspath(os.path.dirname(__file__))

# load the campaign
campaign = es.Campaign(load_state=True)
# load the training data
data_frame_ref = campaign.load_hdf5_data()
# load the prediction 
data_frame_qsn = campaign.load_hdf5_data()

# load reference data
X_ref = data_frame_ref['X_n']
r_ref = data_frame_ref['r_n']

# load data of QSN surrogate
X_qsn = data_frame_qsn['X_n']
r_qsn = data_frame_qsn['r_n']

# Create a QSN analysis object
analysis = es.analysis.QSN_analysis(campaign.surrogate)

#############
# Plot PDFs #
#############

burn = 0; subsample = 10
fig = plt.figure(figsize=[8, 4])
ax = fig.add_subplot(121, xlabel=r'$X_k$')
# X_dom_surr, X_pdf_surr = analysis.get_pdf(X_qsn[burn:-1:subsample].flatten())
# X_dom, X_pdf = analysis.get_pdf(X_ref[burn:-1:subsample].flatten())
X_dom_surr, X_pdf_surr = analysis.get_pdf(X_qsn[:, 0])
X_dom, X_pdf = analysis.get_pdf(X_ref[:, 0])
ax.plot(X_dom, X_pdf, 'k+', label='L96')
ax.plot(X_dom_surr, X_pdf_surr, label='QSN')
plt.yticks([])
plt.legend(loc=0)

ax = fig.add_subplot(122, xlabel=r'$B_k$')
r_dom_surr, r_pdf_surr = analysis.get_pdf(r_qsn[burn:-1:subsample].flatten())
r_dom, r_pdf = analysis.get_pdf(r_ref[burn:-1:subsample].flatten())
ax.plot(r_dom, r_pdf, 'k+', label='L96')
ax.plot(r_dom_surr, r_pdf_surr, label='QSN')
plt.yticks([])
plt.legend(loc=0)

plt.tight_layout()

#############
# Plot ACFs #
#############

acf_lag = 1000

fig = plt.figure(figsize=[8, 4])
ax1 = fig.add_subplot(121, ylabel=r'$\mathrm{ACF}\;X_n$', xlabel='time')
ax2 = fig.add_subplot(122, ylabel=r'$\mathrm{ACF}\;r_n$', xlabel='time')

acf_X_ref = np.zeros(acf_lag - 1)
acf_X_sol = np.zeros(acf_lag - 1)
acf_r_ref = np.zeros(acf_lag - 1)
acf_r_sol = np.zeros(acf_lag - 1)

# average over all spatial points
K = 18
dt = 0.01
for k in range(K):
    print('k=%d' % k)
    acf_X_ref += 1 / K * analysis.auto_correlation_function(X_ref[burn:, k], max_lag=acf_lag)
    acf_X_sol += 1 / K * analysis.auto_correlation_function(X_qsn[burn:, k], max_lag=acf_lag)

    acf_r_ref += 1 / K * analysis.auto_correlation_function(r_ref[burn:, k], max_lag=acf_lag)
    acf_r_sol += 1 / K * analysis.auto_correlation_function(r_qsn[burn:, k], max_lag=acf_lag)

dom_acf = np.arange(acf_lag - 1) * dt
ax1.plot(dom_acf, acf_X_ref, 'k+', label='L96')
ax1.plot(dom_acf, acf_X_sol, label='QSN')
leg = plt.legend(loc=0)

ax2.plot(dom_acf, acf_r_ref, 'k+', label='L96')
ax2.plot(dom_acf, acf_r_sol, label='QSN')
leg = plt.legend(loc=0)

plt.tight_layout()

# #############
# # Plot CCFs #
# #############

# fig = plt.figure(figsize=[8, 4])
# ax1 = fig.add_subplot(121, ylabel=r'$\mathrm{CCF}\;X_n$', xlabel='time')
# ax2 = fig.add_subplot(122, ylabel=r'$\mathrm{CCF}\;r_n$', xlabel='time')

# ccf_X_ref = np.zeros(acf_lag - 1)
# ccf_X_sol = np.zeros(acf_lag - 1)
# ccf_B_ref = np.zeros(acf_lag - 1)
# ccf_B_sol = np.zeros(acf_lag - 1)

# # average over all spatial points
# for k in range(K - 1):
#     print('k=%d' % k)

#     ccf_X_ref += 1 / K * \
#         analysis.cross_correlation_function(X_ref[start_idx:, k], X_ref[start_idx:, k + 1], max_lag=1000)
#     ccf_X_sol += 1 / K * \
#         analysis.cross_correlation_function(X_qsn[start_idx:, k], X_qsn[start_idx:, k + 1], max_lag=1000)

#     ccf_B_ref += 1 / K * \
#         analysis.cross_correlation_function(B_ref[start_idx:, k], B_ref[start_idx:, k + 1], max_lag=1000)
#     ccf_B_sol += 1 / K * \
#         analysis.cross_correlation_function(B_qsn[start_idx:, k], B_qsn[start_idx:, k + 1], max_lag=1000)

# dom_ccf = np.arange(acf_lag - 1) * dt
# ax1.plot(dom_ccf, ccf_X_ref, 'k+', label='L96')
# ax1.plot(dom_ccf, ccf_X_sol, label='QSN')

# ax2.plot(dom_ccf, ccf_B_ref, 'k+', label='L96')
# ax2.plot(dom_ccf, ccf_B_sol, label='QSN')
# leg = plt.legend(loc=0)

# plt.tight_layout()
