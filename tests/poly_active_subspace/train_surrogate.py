def get_poly_training_data(n_mc, D):

    theta = np.random.rand(n_mc, D)
    f = np.zeros(n_mc)

    for k in range(n_mc):
        sol = 1.0
        for i in range(D):
            sol *= 3 * a[i] * theta[k][i]**2 + 1.0
        f[k] = sol / 2**D
        
    ref_mean = np.prod(a+1)/2**D
    ref_std = np.sqrt(np.prod(9*a**2/5 + 2*a + 1)/2**(2*D) - ref_mean**2)

    return theta, f, ref_mean, ref_std

def get_g_func_training_data(n_mc, D):
    theta = np.random.rand(n_mc, D)
    f = np.zeros(n_mc)
    for k in range(n_mc):
        sol = 1.0
        for i in range(D):
            sol *= 2.0 * (np.abs(4.0 * theta[k][i] - 2.0) + a[i]) / (1.0 + a[i])
        f[k] = sol
    return theta, f, 0, 0

#Compute the analytic sobol indices for the test function of sobol_model.py
def exact_sobols_g_function():
    V_i = np.zeros(D)

    for i in range(D):
        V_i[i] = 1.0 / (3.0 * (1 + a[i])**2)

    V = np.prod(1 + V_i) - 1

    print('----------------------')
    print('Exact 1st-order Sobol indices: ', V_i / V)
    
    return V_i/V

def plot_3D_convex_hull(points, ax):

    hull = ConvexHull(points)
    
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]

        surf = ax.plot_trisurf(np.array([p1[0], p2[0], p3[0]]), 
                        np.array([p1[1], p2[1], p3[1]]),
                        np.array([p1[2], p2[2], p3[2]]), 
                        alpha=0.5, color='coral',
                        label=r'Convex hull active subspace')
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d

import easysurrogate as es
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

#active subspace and parameter space dimenions
d = 4
D = 10

#make only first variables important
# a = np.array([1/(2**(i+1)) for i in range(D)])
# a = np.zeros(D)
# a[0] = 1.0; a[1] = 0.0

#Sobol g-func coefficients
a = [2**i for i in range(D-1)]
a.insert(0, 0)

n_mc = 40000
params, samples, ref_mean, ref_std = get_g_func_training_data(n_mc, D)

#save part of the data for testing
test_frac = 0.5
I = np.int(samples.shape[0] * (1.0 - test_frac))

surrogate = es.methods.DAS_network(params[0:I], samples[0:I], d, n_layers=4, n_neurons=200,
                                   save=False, bias=True, alpha=0.001, batch_size=128)
# surrogate = es.methods.ANN(params[0:I], samples[0:I], n_layers=4, n_neurons=200,
                                   # save=False, bias=True, alpha=0.001, batch_size=128)
# train the surrogate on the data
n_iter = 20000
surrogate.train(n_iter, store_loss=True)

# run the trained model forward at training locations
n_mc = samples.shape[0]
pred = np.zeros([I, 1])
y = np.zeros([I, d])
for i in range(I):
    feat_i = (params[i] - surrogate.X_mean) / surrogate.X_std
    pred[i] = surrogate.feed_forward(feat_i.reshape([1, -1]))[0][0]
    y[i] = surrogate.layers[1].h[0:d].flatten()
data = surrogate.y.reshape([-1, 1])

#plot the active subspace
plt.close('all')
if d == 1:
    fig = plt.figure()
    ax = plt.subplot(111, xlabel=r'${\bf y} = W^T{\bf x}$',
                     title='Link function g(y) in active subspace')
    ax.plot(y, pred[:, -1], 's', label='prediction g(y)', alpha=0.5)
    ax.plot(y, data[:, -1], '+', label='data')
    plt.legend()

else:
    fig = plt.figure(figsize=[8, 4])
    ax1 = fig.add_subplot(121, projection='3d',
                          xlabel=r'$y_1$', ylabel=r'$y_2$',
                          title='Link function g(y) in active subspace')
    if d >= 2:
    #     ax1.plot_trisurf(y[:, 0], y[:, 1], pred[:, -1], alpha=0.5)
    # else:
        points = np.array([y[:, 0], y[:, 1], pred[:, -1]]).T
        plot_3D_convex_hull(points, ax1)
    
    ax1.plot(y[:, 0], y[:, 1], data.flatten(), 'o', markersize=3, label=r'data')
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    ax2 = fig.add_subplot(122, xlabel=r'$y_1$', ylabel=r'$y_2$',
                          title='Sampling plan in active subspace')
    ax2.plot(y[:, 0], y[:, 1], 'o')

plt.tight_layout()

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
ax = fig.add_subplot(111, yticks=[], xlabel=r'$f({\bf x}),\;\;g({\bf y})$')
ax.plot(dom_ref, pdf_ref, '--', label='reference')
ax.plot(dom_das, pdf_das, label='deep active subspace')
leg = ax.legend(loc=0)
leg.set_draggable(True)
plt.tight_layout()

exact_sobols_g_function()

if d == 1:
    inputs = np.array(['x%d' % (i+1) for i in range(D)])
    idx = np.argsort(np.abs(surrogate.layers[1].W.T))
    print('Parameters ordered from most to least important:')
    print(np.fliplr((inputs[idx])))
    
surrogate.set_batch_size(1)
surrogate.jacobian(surrogate.X[0].reshape([1,-1]))
f_grad_y = surrogate.layers[1].delta_hy
f_grad_x = surrogate.layers[0].delta_hy
mean_f_grad_y2 = f_grad_y**2
mean_f_grad_x2 = f_grad_x**2
var = np.zeros(d)
var2 = np.zeros(D)
analysis = es.analysis.BaseAnalysis()

for i in range(1, surrogate.X.shape[0]):
    surrogate.jacobian(surrogate.X[i].reshape([1,-1]))
    f_grad_y2 = surrogate.layers[1].delta_hy**2
    f_grad_x2 = surrogate.layers[0].delta_hy**2
    mean_f_grad_y2, var = analysis.recursive_moments(f_grad_y2, mean_f_grad_y2, var, i)
    mean_f_grad_x2, var2 = analysis.recursive_moments(f_grad_x2, mean_f_grad_x2, var2, i)

inputs = np.array(['x%d' % (i+1) for i in range(D)])
idx = np.argsort(np.abs(mean_f_grad_x2).T)
print('Parameters ordered from most to least important:')
print(np.fliplr((inputs[idx])))