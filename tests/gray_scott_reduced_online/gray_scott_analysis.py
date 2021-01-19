import os
import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es

home = os.path.abspath(os.path.dirname(__file__))

# create a campaign
campaign = es.Campaign()

# use a the base analysis here
analysis = es.analysis.BaseAnalysis()

# load the training data (from gray_scott_rk4.py)
data_frame_ref = campaign.load_hdf5_data(file_path='../samples/gray_scott_training_stationary.hdf5')
# load the data from the online trained reduced model
data_frame_red = campaign.load_hdf5_data(file_path='../samples/gray_scott_online_M1_tau_infty2.hdf5')
# load the data from the wrong model
data_frame_wrong = campaign.load_hdf5_data(file_path='../samples/gray_scott_training_stationary_epsilon.hdf5')
# load the data from the unparameterised model
data_frame_unp = campaign.load_hdf5_data(file_path='../samples/gray_scott_unparam_stationary.hdf5')

# load reference data
Q_ref = data_frame_ref['Q_HR']
# load data of reduced surrogate
Q_reduced = data_frame_red['Q_LR']
# load data of unparam solution
Q_unparam = data_frame_unp['Q_LR']
# load data of wrong solution
Q_wrong = data_frame_wrong['Q_LR']

dQ_ref = data_frame_ref['Q_HR'] - data_frame_ref['Q_LR']
dQ_red = data_frame_red['Q_HR'] - data_frame_red['Q_LR']
dQ_wrong = data_frame_wrong['Q_HR'] - data_frame_wrong['Q_LR']

#############
# Plot pdfs #
#############

start_idx = 0
fig = plt.figure(figsize=[12, 4])
for i in range(4):
    ax = fig.add_subplot(1, 4, i + 1, xlabel=r'$Q_%d$' % (i + 1))
    dom_surr, pdf_surr = analysis.get_pdf(Q_reduced[start_idx:-1:1, i])
    dom_ref, pdf_ref = analysis.get_pdf(Q_ref[start_idx:-1:1, i])
    dom_unparam, pdf_unparam = analysis.get_pdf(Q_unparam[start_idx:-1:1, i])

    ax.plot(dom_ref, pdf_ref, 'k+', label='Ref')
    ax.plot(dom_surr, pdf_surr, label='Reduced')
    ax.plot(dom_unparam, pdf_unparam, ':', label='Unparam')

plt.yticks([])
plt.legend(loc=0)

plt.tight_layout()

# Plot Q_i vs dQ
fig = plt.figure(figsize=[12,4])

for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xlabel=r'$Q_%d$' % (i+1), ylabel=r'$\Delta Q_%d$' % (i+1))
    ax.plot(Q_ref[:, i], dQ_ref[:, i], '.', label='reference')
    ax.plot(Q_reduced[:, i], dQ_red[:, i], '.', label='reduced')
    ax.plot(Q_wrong[:, i], dQ_wrong[:, i], '.', label='wrong')

plt.legend(loc=0)

plt.tight_layout()

plt.show()