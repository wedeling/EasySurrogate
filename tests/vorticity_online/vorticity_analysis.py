import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es

# create a campaign
campaign = es.Campaign()

# use a the base analysis here
analysis = es.analysis.BaseAnalysis()

# load the training data (from gray_scott_rk4.py)
data_frame = campaign.load_hdf5_data(file_path='/home/wouter/VECMA/samples/reduced_vorticity_online_N2_M20.hdf5')

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

fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
ax.set_xlabel('days')
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

ax.plot(data_frame['Q_HR'][0:-1:630,0], 'o', label=r'$E^{ref}$')
ax.plot(data_frame['Q_LR'][0:-1:630,0], label=r'$E$')
ax.set_ylim([0.0003, 0.0005])
ax.tick_params(axis='y', labelcolor=colors[1])
ax.set_ylabel('E')
leg = ax.legend(loc=4)
leg.set_draggable(True)

ax2.plot(data_frame['Q_HR'][0:-1:630,1], 'o', color=colors[2], label=r'$Z^{ref}$')
ax2.plot(data_frame['Q_LR'][0:-1:630,1], color=colors[3], label=r'$Z$')
ax2.tick_params(axis='y', labelcolor=colors[3])
ax2.set_ylim([0, 0.012])
ax2.set_ylabel('Z')
leg2 = ax2.legend(loc=0)
leg2.set_draggable(True)

plt.tight_layout()
plt.show()