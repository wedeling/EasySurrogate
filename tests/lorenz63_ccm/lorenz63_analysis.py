import os
import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es

dt = 0.01

home = os.path.abspath(os.path.dirname(__file__))

campaign = es.Campaign(load_state=True)
data_frame_ref = campaign.load_hdf5_data()
data_frame_qsn = campaign.load_hdf5_data()

#load reference data
X_ref = data_frame_ref['X']
Y_ref = data_frame_ref['Y']

#load data of ccm surrogate
X_ccm = data_frame_qsn['X']
Y_ccm = data_frame_qsn['Y']

#create ccm analysis object
analysis = es.analysis.QSN_analysis()

#############   
# Plot PDEs #
#############

start_idx = 0
fig = plt.figure(figsize=[8,4])
ax = fig.add_subplot(121, xlabel=r'$X$')
X_dom_surr, X_pde_surr = analysis.get_pdf(X_ccm[start_idx:-1:10].flatten())
X_dom, X_pde = analysis.get_pdf(X_ref[start_idx:-1:10].flatten())
ax.plot(X_dom, X_pde, 'k+', label='L96')
ax.plot(X_dom_surr, X_pde_surr, label='ccm')
plt.yticks([])
plt.legend(loc=0)

ax = fig.add_subplot(122, xlabel=r'$Y$')
Y_dom_surr, Y_pde_surr = analysis.get_pdf(Y_ccm[start_idx:-1:10].flatten())
Y_dom, Y_pde = analysis.get_pdf(Y_ref[start_idx:-1:10].flatten())
ax.plot(Y_dom, Y_pde, 'k+', label='L96')
ax.plot(Y_dom_surr, Y_pde_surr, label='ccm')
plt.yticks([])
plt.legend(loc=0)

plt.tight_layout()

#############   
# Plot ACFs #
#############

acf_lag = 1000

fig = plt.figure(figsize=[8,4])
ax1 = fig.add_subplot(121, ylabel='$\mathrm{ACF}\;X$', xlabel='time')
ax2 = fig.add_subplot(122, ylabel='$\mathrm{ACF}\;Y$', xlabel='time')

acf_X_ref = np.zeros(acf_lag-1); acf_X_sol = np.zeros(acf_lag-1)
acf_Y_ref = np.zeros(acf_lag-1); acf_Y_sol = np.zeros(acf_lag-1)

#average over all spatial points
acf_X_ref = analysis.auto_correlation_function(X_ref[start_idx:], max_lag=acf_lag)
acf_X_sol = analysis.auto_correlation_function(X_ccm[start_idx:], max_lag=acf_lag)

acf_Y_ref = analysis.auto_correlation_function(Y_ref[start_idx:], max_lag=acf_lag)
acf_Y_sol = analysis.auto_correlation_function(Y_ccm[start_idx:], max_lag=acf_lag)

dom_acf = np.arange(acf_lag-1)*dt
ax1.plot(dom_acf, acf_X_ref, 'k+', label='L96')
ax1.plot(dom_acf, acf_X_sol, label='ccm')
leg = plt.legend(loc=0)

ax2.plot(dom_acf, acf_Y_ref, 'k+', label='L96')
ax2.plot(dom_acf, acf_Y_sol, label='ccm')
leg = plt.legend(loc=0)

plt.tight_layout()

#############   
# Plot CCFs #
#############

fig = plt.figure(figsize=[4,4])
ax1 = fig.add_subplot(111, ylabel='$\mathrm{CCF(X,Y)}$', xlabel='time')

ccf_ref = analysis.cross_correlation_function(X_ref[start_idx:], Y_ref[start_idx:], max_lag=1000)
ccf_sol = analysis.cross_correlation_function(X_ccm[start_idx:], Y_ccm[start_idx:], max_lag=1000)

dom_ccf = np.arange(acf_lag - 1)*dt
ax1.plot(dom_ccf, ccf_ref, 'k+', label='L96')
ax1.plot(dom_ccf, ccf_sol, label='ccm')

leg = plt.legend(loc=0)

plt.tight_layout()