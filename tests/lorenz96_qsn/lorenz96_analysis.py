import os
import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es

home = os.path.abspath(os.path.dirname(__file__))

# load the campaign
campaign = es.Campaign(load_state=True)
# load the training data (from lorenz96.py)
data_frame_ref = campaign.load_hdf5_data()
# load the data from lorenz96_qsn.py here
data_frame_qsn = campaign.load_hdf5_data()

# load reference data
X_ref = data_frame_ref['X_data']
B_ref = data_frame_ref['B_data']

# load data of QSN surrogate
X_qsn = data_frame_qsn['X_data']
B_qsn = data_frame_qsn['B_data']

# create QSN analysis object
analysis = es.analysis.QSN_analysis(campaign.surrogate)

#############
# Plot PDFs #
#############

start_idx = 0
fig = plt.figure(figsize=[8, 4])
ax = fig.add_subplot(121, xlabel=r'$X_k$')
X_dom_surr, X_pde_surr = analysis.get_pdf(X_qsn[start_idx:-1:10].flatten())
X_dom, X_pde = analysis.get_pdf(X_ref[start_idx:-1:10].flatten())
ax.plot(X_dom, X_pde, 'k+', label='L96')
ax.plot(X_dom_surr, X_pde_surr, label='QSN')
plt.yticks([])
plt.legend(loc=0)

ax = fig.add_subplot(122, xlabel=r'$B_k$')
B_dom_surr, B_pde_surr = analysis.get_pdf(B_qsn[start_idx:-1:10].flatten())
B_dom, B_pde = analysis.get_pdf(B_ref[start_idx:-1:10].flatten())
ax.plot(B_dom, B_pde, 'k+', label='L96')
ax.plot(B_dom_surr, B_pde_surr, label='QSN')
plt.yticks([])
plt.legend(loc=0)

plt.tight_layout()
plt.savefig('pdf.png')

# #############
# # Plot ACFs #
# #############

# acf_lag = 1000

# fig = plt.figure(figsize=[8, 4])
# ax1 = fig.add_subplot(121, ylabel=r'$\mathrm{ACF}\;X_n$', xlabel='time')
# ax2 = fig.add_subplot(122, ylabel=r'$\mathrm{ACF}\;r_n$', xlabel='time')

# acf_X_ref = np.zeros(acf_lag - 1)
# acf_X_sol = np.zeros(acf_lag - 1)
# acf_B_ref = np.zeros(acf_lag - 1)
# acf_B_sol = np.zeros(acf_lag - 1)

# # average over all spatial points
# K = 18
# dt = 0.01
# for k in range(K):
#     print('k=%d' % k)
#     acf_X_ref += 1 / K * analysis.auto_correlation_function(X_ref[start_idx:, k], max_lag=acf_lag)
#     acf_X_sol += 1 / K * analysis.auto_correlation_function(X_qsn[start_idx:, k], max_lag=acf_lag)

#     acf_B_ref += 1 / K * analysis.auto_correlation_function(B_ref[start_idx:, k], max_lag=acf_lag)
#     acf_B_sol += 1 / K * analysis.auto_correlation_function(B_qsn[start_idx:, k], max_lag=acf_lag)

# dom_acf = np.arange(acf_lag - 1) * dt
# ax1.plot(dom_acf, acf_X_ref, 'k+', label='L96')
# ax1.plot(dom_acf, acf_X_sol, label='QSN')
# leg = plt.legend(loc=0)

# ax2.plot(dom_acf, acf_B_ref, 'k+', label='L96')
# ax2.plot(dom_acf, acf_B_sol, label='QSN')
# leg = plt.legend(loc=0)

# plt.tight_layout()

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
