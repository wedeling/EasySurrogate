import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es

# create a campaign
campaign = es.Campaign()

# use a the base analysis here
analysis = es.analysis.BaseAnalysis()

# load the training data (from gray_scott_rk4.py)
data_frame = campaign.load_hdf5_data(file_path='../samples/reduced_vorticity_online_N2_M1.hdf5')

# load reference data
Q_ref = data_frame['Q_HR']

# load data of reduced surrogate
Q_reduced = data_frame['Q_LR']

#############
# Plot pdfs #
#############

start_idx = 0
fig = plt.figure(figsize=[8, 4])
for i in range(2):
    ax = fig.add_subplot(1, 2, i + 1, xlabel=r'$Q_%d$' % (i + 1))
    dom_surr, pdf_surr = analysis.get_pdf(Q_reduced[start_idx:-1:10, i])
    dom_ref, pdf_ref = analysis.get_pdf(Q_ref[start_idx:-1:10, i])
    ax.plot(dom_ref, pdf_ref, 'k+', label='Ref')
    ax.plot(dom_surr, pdf_surr, label='Reduced')
plt.yticks([])
plt.legend(loc=0)

plt.tight_layout()
