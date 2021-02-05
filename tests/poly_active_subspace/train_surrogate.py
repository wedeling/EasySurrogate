def get_training_data(n_mc, D):

    theta = np.random.rand(n_mc, D)
    f = np.zeros(n_mc)

    for k in range(n_mc):
        sol = 1.0
        for i in range(D):
            sol *= 3 * a[i] * theta[k][i]**2 + 1.0
        f[k] = sol / 2**D

    return theta, f

import easysurrogate as es
import matplotlib.pyplot as plt
import numpy as np

#active subspace and parameter space dimenions
d = 2
D = 5

#make only first variables important
# a = np.array([1/(2*(i+1)) for i in range(D)])
a = np.zeros(D)
a[0] = 1.0; a[1] = 1.0
n_mc = 1000
params, samples = get_training_data(n_mc, D)

#save part of the data for testing
test_frac = 0.5
I = np.int(samples.shape[0] * (1.0 - test_frac))

surrogate = es.methods.DAS_network(params[0:I], samples[0:I], d, n_layers=4, n_neurons=50,
                                   save=False, bias=True, alpha=0.001, batch_size=128)
# train the surrogate on the data
n_iter = 10000
surrogate.train(n_iter, store_loss=True)

# run the trained model forward at training locations
n_mc = samples.shape[0]
pred = np.zeros(n_mc)
y = np.zeros([n_mc, d])
for i in range(I):
    feat_i = (params[i] - surrogate.X_mean) / surrogate.X_std
    pred[i] = surrogate.feed_forward(feat_i.reshape([1, -1]))[0][0]
    y[i] = surrogate.layers[1].h.flatten()
data = (samples.flatten() - surrogate.y_mean) / surrogate.y_std

#plot the active subspace
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
ref_mean = np.prod(a+1)/2**D
ref_std = np.sqrt(np.prod(9*a**2/5 + 2*a + 1)/2**(2*D) - ref_mean**2)

#run the model forward at test locations
pred = np.zeros(n_mc-I)
idx = 0
for i in range(I, n_mc):
    feat_i = (params[i] - surrogate.X_mean) / surrogate.X_std
    pred_i = surrogate.feed_forward(feat_i.reshape([1, -1]))[0][0]
    pred[idx] = pred_i * surrogate.y_std + surrogate.y_mean
    idx += 1 

das_mean = np.mean(pred)
das_std = np.std(pred)
print("--------------------------------------")
print("Analytic mean = %.4e" % ref_mean)
print("Computed mean = %.4e" % das_mean)
print("--------------------------------------")
print("Analytic standard deviation = %.4e" % ref_std)
print("Computed standard deviation = %.4e" % das_std)
print("--------------------------------------")

#perform some basic analysis
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
