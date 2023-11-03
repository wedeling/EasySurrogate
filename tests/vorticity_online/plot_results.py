import matplotlib.pyplot as plt
import easysurrogate as es

campaign = es.Campaign()
analysis = es.analysis.BaseAnalysis()

data = campaign.load_hdf5_data()

Q_HR_1 = data['Q_HR'][:, 0]
Q_HR_2 = data['Q_HR'][:, 1]
Q_LR_1 = data['Q_LR'][:, 0]
Q_LR_2 = data['Q_LR'][:, 1]

dom_HR_1, pdf_HR_1 = analysis.get_pdf(Q_HR_1)
dom_HR_2, pdf_HR_2 = analysis.get_pdf(Q_HR_2)
dom_LR_1, pdf_LR_1 = analysis.get_pdf(Q_LR_1)
dom_LR_2, pdf_LR_2 = analysis.get_pdf(Q_LR_2)

fig = plt.figure(figsize=[8, 4])

ax = fig.add_subplot(121)
ax.plot(dom_HR_1, pdf_HR_1, '--k')
ax.plot(dom_LR_1, pdf_LR_1)

ax = fig.add_subplot(122)
ax.plot(dom_HR_2, pdf_HR_2, '--k')
ax.plot(dom_LR_2, pdf_LR_2)

plt.tight_layout()
plt.show()
