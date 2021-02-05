import easysurrogate as es
import easyvvuq as uq
from custom import CustomEncoder
import matplotlib.pyplot as plt
import numpy as np


def get_training_data(n_mc, D):

    theta = np.random.rand(n_mc, D)
    f = np.zeros(n_mc)

    for k in range(n_mc):
        sol = 1.0
        for i in range(D):
            sol *= 3 * a[i] * theta[k][i]**2 + 1.0
        f[k] = sol / 2**D

    return theta, f


# load campaign
campaign = uq.Campaign(work_dir='~/VECMA/Campaigns', state_file='campaign3.json')
sampler = campaign.get_active_sampler()
sampler.load_state("sampler_campaign3.pickle")

# load data frame, input and output names
data_frame = campaign.get_collation_result()
inputs = list(sampler.vary.get_keys())
qoi_cols = ['cumDeath']
samples = []
for k in qoi_cols:
    run_id_int = [int(run_id.split('Run_')[-1]) for run_id in data_frame[k].keys()]
    for run_id in range(1, np.max(run_id_int) + 1):
        samples.append(data_frame[k]['Run_' + str(run_id)])
params = sampler.xi_d
samples = np.array(samples)[:, -1]

# qoi_cols = ['f']

# #extract training data from EasyVVUQ data frame
# params = []; samples = []
# for run_id in data_frame[('run_id', 0)].unique():
#     for k in qoi_cols:
#         xi = data_frame.loc[data_frame[('run_id', 0)] == run_id][inputs].values
#         values = data_frame.loc[data_frame[('run_id', 0)] == run_id][k].values
#         params.append(xi.flatten())
#         samples.append(values.flatten())

# params = np.array(params)
# samples = np.array(samples)

# surrogate = es.methods.ANN_Surrogate()
# n_iter = 20000
# surrogate.train(params, samples, n_iter, bias=False)

D = params.shape[1]
d = 5
# n_mc = 1000
# #make only first variables important
# a = np.array([1/(2*(i+1)) for i in range(D)])
# # a = np.zeros(D)
# # a[0] = 1.0; a[1] = 1.0

# params, samples = get_training_data(n_mc, D)
test_frac = 0.0
I = np.int(samples.shape[0] * (1.0 - test_frac))

surrogate = es.methods.DAS_network(params[0:I], samples[0:I], d, n_layers=4, n_neurons=50,
                                   save=False, bias=True, alpha=0.001, batch_size=128)
# train the surrogate on the data
n_iter = 10000
surrogate.train(n_iter, store_loss=True)

n_mc = samples.shape[0]
pred = np.zeros(n_mc)
y = np.zeros([n_mc, d])
for i in range(n_mc):
    feat_i = (params[i] - surrogate.X_mean) / surrogate.X_std
    pred[i] = surrogate.feed_forward(feat_i.reshape([1, -1]))[0][0]
    y[i] = surrogate.layers[1].h.flatten()

data = (samples.flatten() - surrogate.y_mean) / surrogate.y_std

plt.close('all')
if d == 1:
    fig = plt.figure()
    ax = plt.subplot(111, xlabel=r'${\bf y} = W^T\xi$',
                     title='Link function g(y) in active subspace')
    ax.plot(y, data, '+', label='data')
    ax.plot(y, pred, 's', label='prediction g(y)', alpha=0.5)
    plt.legend()

else:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=[8, 4])
    ax1 = fig.add_subplot(121, projection='3d',
                          xlabel=r'$y_1$', ylabel=r'$y_2$',
                          title='Link function g(y) in active subspace')
    ax1.plot_trisurf(y[:, 0], y[:, 1], pred, alpha=0.5)
    ax1.plot(y[:, 0], y[:, 1], data, 'o', label=r'data')
    plt.legend()
    ax2 = fig.add_subplot(122, xlabel=r'$y_1$', ylabel=r'$y_2$',
                          title='Sampling plan in active subspace')
    ax2.plot(y[:, 0], y[:, 1], 'o')

plt.tight_layout()

# analytic mean and standard deviation
# ref_mean = np.prod(a+1)/2**D
# ref_std = np.sqrt(np.prod(9*a**2/5 + 2*a + 1)/2**(2*D) - ref_mean**2)

pred = np.zeros(n_mc)
# params_pred = np.random.rand(n_mc, D)
params_pred = params
for i in range(n_mc):
    feat_i = (params_pred[i] - surrogate.X_mean) / surrogate.X_std
    pred_i = surrogate.feed_forward(feat_i.reshape([1, -1]))[0][0]
    pred[i] = pred_i * surrogate.y_std + surrogate.y_mean

# das_mean = np.mean(pred)
# das_std = np.std(pred)
# print("--------------------------------------")
# print("Analytic mean = %.4e" % ref_mean)
# print("Computed mean = %.4e" % das_mean)
# print("--------------------------------------")
# print("Analytic standard deviation = %.4e" % ref_std)
# print("Computed standard deviation = %.4e" % das_std)
# print("--------------------------------------")

analysis = es.analysis.BaseAnalysis()
analysis.get_pdf(samples, 100)
dom_ref, pdf_ref = analysis.get_pdf(samples, 100)
dom_das, pdf_das = analysis.get_pdf(pred, 100)

fig = plt.figure()
ax = fig.add_subplot(111, yticks=[])
ax.plot(dom_ref, pdf_ref, label='reference')
ax.plot(dom_das, pdf_das, label='deep active subspace')
leg = ax.legend(loc=0)
leg.set_draggable(True)
plt.tight_layout()
