import os
import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es

home = os.path.abspath(os.path.dirname(__file__))

#create a campaign
campaign = es.Campaign()

#use a the base analysis here
analysis = es.analysis.BaseAnalysis()

#load the training data (from gray_scott_rk4.py)
data_frame_ref = campaign.load_hdf5_data(file_path = home + '/samples/gray_scott_reference.hdf5')
#load the data from the trained reduced model
data_frame_red = campaign.load_hdf5_data(file_path = home + '/samples/gray_scott_model.hdf5')

# load reference data
Q_ref  = data_frame_ref['Q_HF']

# load data of reduced surrogate
Q_reduced = data_frame_red['Q_HF']

#############
# Plot pdfs #
#############

start_idx = 0
fig = plt.figure(figsize=[12, 4])
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xlabel=r'$Q_%d$' % (i+1))
    dom_surr, pdf_surr = analysis.get_pdf(Q_reduced[start_idx:-1:1, i])
    dom_ref, pdf_ref = analysis.get_pdf(Q_ref[start_idx:-1:1, i])
    ax.plot(dom_ref, pdf_ref, 'k+', label='Ref')
    ax.plot(dom_surr, pdf_surr, label='Reduced')
plt.yticks([])
plt.legend(loc=0)

plt.tight_layout()