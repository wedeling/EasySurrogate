import os
import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es

#home = os.path.abspath(os.path.dirname(__file__))

# load the campaign
campaign = es.Campaign(load_state=True)
# load the training data (from lorenz96.py)
data_frame_ref = campaign.load_hdf5_data()
# load the data from lorenz96_surrogate.py here
data_frame_surr = campaign.load_hdf5_data()

# load reference data
X_ref = data_frame_ref['X_data']
B_ref = data_frame_ref['B_data']

# load data of QSN surrogate
X_surr = data_frame_surr['X_data']
B_surr = data_frame_surr['B_data']

# create analysis object
analysis = es.analysis.BaseAnalysis()

#############
# Plot PDFs #
#############

start_idx = 0
fig = plt.figure(figsize=[8, 4])
ax = fig.add_subplot(121, xlabel=r'$X_n$')
X_dom_surr, X_pde_surr = analysis.get_pdf(X_surr[start_idx:-1:10].flatten())
X_dom, X_pde = analysis.get_pdf(X_ref[start_idx:-1:10].flatten())
ax.plot(X_dom, X_pde, 'k+', lw=2, label='L96')
ax.plot(X_dom_surr, X_pde_surr, lw=2, label='Surr')
plt.yticks([])
plt.legend(loc=0)

ax = fig.add_subplot(122, xlabel=r'$\tilde{r}_n$')
B_dom_surr, B_pde_surr = analysis.get_pdf(B_surr[start_idx:-1:10].flatten())
B_dom, B_pde = analysis.get_pdf(B_ref[start_idx:-1:10].flatten())
ax.plot(B_dom, B_pde, 'k+', lw=2, label='L96')
ax.plot(B_dom_surr, B_pde_surr, lw=2, label='Surr')
plt.yticks([])
plt.legend(loc=0)

plt.tight_layout()
