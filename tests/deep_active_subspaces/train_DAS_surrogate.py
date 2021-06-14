import easyvvuq as uq
import easysurrogate as es
import numpy as np
import matplotlib.pyplot as plt

# work directory
WORK_DIR = '/tmp'

# location of the EasyVVUQ database
ID = 'g_func'
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
plt.tricontour(params[:,0], params[:, 1], samples.flatten(), 50)

# input dimension (15)
D = params.shape[1]

# assumed dimension active subspace
d = 1

surrogate = es.methods.DAS_Surrogate()
surrogate.train(params, samples, 
                d, n_iter=10000, n_layers=4, n_neurons=100, 
                test_frac = 0.2)
dims = surrogate.get_dimensions()

# plot contour if problem is two-dimensional
if D == 2:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.tricontour(params[:, 0], params[:, 1], samples.flatten(), 50)
    # Extract the coordinate vector of the 1D active subspace
    # This is the weight vector of the 2nd layer
    w = surrogate.neural_net.layers[1].W.flatten()
    # if both entries are < 0, make both positive for plotting purpose
    if (np.sign(w) == [-1, -1]).all():
        w = np.abs(w)
    # plot the coordinate vector
    plt.arrow(0, 0, w[0], w[1])

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

#perform some basic analysis
analysis = es.analysis.DAS_analysis(surrogate)

n_mc = 10**4
params = np.array([p.sample(n_mc) for p in sampler.vary.get_values()]).T
idx, mean = analysis.sensitivity_measures(params)
params_ordered = np.array(list(sampler.vary.get_keys()))[idx[0]]

fig = plt.figure('sensitivity', figsize=[4, 8])
ax = fig.add_subplot(111)
ax.set_ylabel(r'$\frac{\partial ||y||^2_2}{\partial x_i}$', fontsize=14)
# find max quad order for every parameter
ax.bar(range(mean.size), height = mean[idx].flatten())
ax.set_xticks(range(mean.size))
ax.set_xticklabels(params_ordered)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()