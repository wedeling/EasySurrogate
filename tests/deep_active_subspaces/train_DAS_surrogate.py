"""
Script to train a Deep Active Subspace on a EasyVVUQ data frame
"""

import easyvvuq as uq
import easysurrogate as es
import numpy as np
import matplotlib.pyplot as plt

# work directory
WORK_DIR = '/tmp'

# location of the EasyVVUQ database
ID = 'func'
DB_LOCATION = "sqlite:///" + WORK_DIR + "/campaign%s.db" % ID

# reload easyvvuq campaign
campaign = uq.Campaign(name=ID, db_location=DB_LOCATION)
print("===========================================")
print("Reloaded campaign {}".format(ID))
print("===========================================")
sampler = campaign.get_active_sampler()
campaign.set_sampler(sampler, update=True)

# name of the quantity of interest, the column name of the output file
output_columns = ["f"]

# create an EasySurrogate campaign
surr_campaign = es.Campaign(name=ID)

# load the training data
params, samples = surr_campaign.load_easyvvuq_data(campaign, output_columns)
samples = samples[output_columns[0]]

# input dimension (15)
D = params.shape[1]
# assumed dimension active subspace
d = 5

# create DAS surrogate object
surrogate = es.methods.DAS_Surrogate()

# train the DAS surrogate
surrogate.train(params, samples, 
                d, n_iter=10000, n_layers=4, n_neurons=100, 
                test_frac = 0.2)

# useful dimensions related to the surrogate
dims = surrogate.get_dimensions()

# plot contours if problem is two-dimensional
if D == 2:
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel=r'$x_1$', ylabel=r'$x_2$')
    ax.tricontour(params[:, 0], params[:, 1], samples.flatten(), 30)
    # Extract the coordinate vector of the 1D active subspace
    # This is the weight vector of the 2nd layer
    w = surrogate.neural_net.layers[1].W.flatten()
    # plot the coordinate vector
    ax.annotate("", xytext=(0, 0), xy=(-w[0], -w[1]),
                arrowprops=dict(width=5, headwidth=10))
    if d == D:
        ax.annotate("", xy=(0, 0), xytext=(w[2], w[3]),
                    arrowprops=dict(width=5, headwidth=10, color='r'))
    plt.axis('equal')
    plt.tight_layout()

#########################
# Compute error metrics #
#########################

# run the trained model forward at training locations
n_mc = dims['n_train']
pred = np.zeros([n_mc, dims['n_out']])
for i in range(n_mc):
    pred[i,:] = surrogate.predict(params[i])
   
train_data = samples[0:dims['n_train']]
rel_err_train = np.linalg.norm(train_data - pred)/np.linalg.norm(train_data)

# run the trained model forward at test locations
pred = np.zeros([dims['n_test'], dims['n_out']])
for idx, i in enumerate(range(dims['n_train'], dims['n_samples'])):
    pred[idx] = surrogate.predict(params[i])
test_data = samples[dims['n_train']:]
rel_err_test = np.linalg.norm(test_data - pred)/np.linalg.norm(test_data)

print('================================')
print('Relative error on training set = %.4f' % rel_err_train)
print('Relative error on test set = %.4f' % rel_err_test)
print('================================')

###########################
# Sensitivity experiments #
###########################

# create DAS analysis object
analysis = es.analysis.DAS_analysis(surrogate)
# draw MC samples from the inputs
n_mc = 10**4
params_mc = np.array([p.sample(n_mc) for p in sampler.vary.get_values()]).T
# evaluate sensitivity integral on sampling plan
idx, mean_grad = analysis.sensitivity_measures(params_mc)
params_ordered = np.array(list(sampler.vary.get_keys()))[idx[0]]

fig = plt.figure('sensitivity', figsize=[4, 8])
ax = fig.add_subplot(111)
ax.set_ylabel(r'$\frac{\partial ||y||^2_2}{\partial x_i}$', fontsize=14)
# find max quad order for every parameter
ax.bar(range(mean_grad.size), height = mean_grad[idx].flatten())
ax.set_xticks(range(mean_grad.size))
ax.set_xticklabels(params_ordered)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()