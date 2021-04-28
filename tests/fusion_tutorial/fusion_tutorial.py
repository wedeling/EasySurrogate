import easyvvuq as uq
import easysurrogate as es

import os
import chaospy as cp
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def write_template(params):
    str = ""
    first = True
    for k in params.keys():
        if first:
            str += '{"%s": "$%s"' % (k,k) ; first = False
        else:
            str += ', "%s": "$%s"' % (k,k)
    str += '}'
    print(str, file=open('fusion.template','w'))

def define_params():
    return {
        "Qe_tot": {"type": "float", "min": 1.0e6, "max": 50.0e6, "default": 2e6},
        "H0": {"type": "float", "min": 0.00, "max": 1.0, "default": 0},
        "Hw": {"type": "float", "min": 0.01, "max": 100.0, "default": 0.1},
        "Te_bc": {"type": "float", "min": 10.0, "max": 1000.0, "default": 100},
        "chi": {"type": "float", "min": 0.01, "max": 100.0, "default": 1},
        "a0": {"type": "float", "min": 0.2, "max": 10.0, "default": 1},
        "R0": {"type": "float", "min": 0.5, "max": 20.0, "default": 3},
        "E0": {"type": "float", "min": 1.0, "max": 10.0, "default": 1.5},
        "b_pos": {"type": "float", "min": 0.95, "max": 0.99, "default": 0.98},
        "b_height": {"type": "float", "min": 3e19, "max": 10e19, "default": 6e19},
        "b_sol": {"type": "float", "min": 2e18, "max": 3e19, "default": 2e19},
        "b_width": {"type": "float", "min": 0.005, "max": 0.025, "default": 0.01},
        "b_slope": {"type": "float", "min": 0.0, "max": 0.05, "default": 0.01},
        "nr": {"type": "integer", "min": 10, "max": 1000, "default": 100},
        "dt": {"type": "float", "min": 1e-3, "max": 1e3, "default": 100},
        "out_file": {"type": "string", "default": "output.csv"}
    }

def define_vary():
    vary_all = {
        "Qe_tot":   cp.Uniform(1.8e6, 2.2e6),
        "H0":       cp.Uniform(0.0,   0.2),
        "Hw":       cp.Uniform(0.1,   0.5),
        "chi":      cp.Uniform(0.8,   1.2),
        "Te_bc":    cp.Uniform(80.0,  120.0),
        "a0":       cp.Uniform(0.9,   1.1),
        "R0":       cp.Uniform(2.7,   3.3),
        "E0":       cp.Uniform(1.4,   1.6),
        "b_pos":    cp.Uniform(0.95,  0.99),
        "b_height": cp.Uniform(5e19,  7e19),
        "b_sol":    cp.Uniform(1e19,  3e19),
        "b_width":  cp.Uniform(0.015, 0.025),
        "b_slope":  cp.Uniform(0.005, 0.020)
    }
    vary_2 =  {
        "Qe_tot":   cp.Uniform(1.8e6, 2.2e6),
        "Te_bc":    cp.Uniform(80.0,  120.0)
    }
    vary_5 =  {
        "Qe_tot":   cp.Uniform(1.8e6, 2.2e6),
        "H0":       cp.Uniform(0.0,   0.2),
        "Hw":       cp.Uniform(0.1,   0.5),
        "chi":      cp.Uniform(0.8,   1.2),
        "Te_bc":    cp.Uniform(80.0,  120.0)
    }
    vary_10 = {
        "Qe_tot":   cp.Uniform(1.8e6, 2.2e6),
        "H0":       cp.Uniform(0.0,   0.2),
        "Hw":       cp.Uniform(0.1,   0.5),
        "chi":      cp.Uniform(0.8,   1.2),
        "Te_bc":    cp.Uniform(80.0,  120.0),
        "b_pos":    cp.Uniform(0.95,  0.99),
        "b_height": cp.Uniform(5e19,  7e19),
        "b_sol":    cp.Uniform(1e19,  3e19),
        "b_width":  cp.Uniform(0.015, 0.025),
        "b_slope":  cp.Uniform(0.005, 0.020)
    }
    return vary_10

def get_inputs(campaign, sampler, n_mc):
    # number of inputs
    D = len(sampler.vary.get_keys())
    # store the parameters in theta, will be used as features
    theta = np.zeros([n_mc, D])
    # loop over all runs
    for i, run in enumerate(campaign.list_runs()):
        # get the paramater values
        values = run[1]['params']
        for j, param in enumerate(values):
            # last entry in values will be the output filename, do not store
            if j < D:
                theta[i, j] = values[param]
    theta = theta.tolist()
    theta = [np.array(x).reshape(-1, 1) for x in theta]
    return theta

def get_outputs(data_frame):
    qoi = 'te'
    samples = []
    for run_id in data_frame[('run_id', 0)].unique():
        values = data_frame.loc[data_frame[('run_id', 0)] == run_id][qoi].values
        samples.append(values.flatten())
    samples = np.array(samples)
    return samples

def derivative_based_sa(surrogate, sampler, D=10):
    # set the batch size to 1
    surrogate.neural_net.set_batch_size(1)

    # get the derivatives
    surrogate.neural_net.jacobian(surrogate.neural_net.X[0].reshape([1, -1]))
    # the derivative of the first layer is the one we want
    f_grad_x = surrogate.neural_net.layers[0].delta_hy
    # square it
    mean_f_grad_x2 = f_grad_x ** 2
    var2 = np.zeros(D)
    # create a basic EasySurrogate analysis class
    analysis = es.analysis.BaseAnalysis()

    # loop over all training samples
    for i in range(1, surrogate.neural_net.X.shape[0]):
        surrogate.neural_net.jacobian(surrogate.neural_net.X[i].reshape([1, -1]))
        f_grad_x2 = surrogate.neural_net.layers[0].delta_hy ** 2
        # use recusive formulas to update the mean (and variance although I don't use it)
        # with a new f_grad_x2 sample
        mean_f_grad_x2, var2 = analysis.recursive_moments(f_grad_x2, mean_f_grad_x2, var2, i)

    # get all input names
    inputs = np.array(list(sampler.vary.get_keys()))
    # sort the sensitivity indices
    idx = np.argsort(np.abs(mean_f_grad_x2).T)
    # print input in order of importance
    print('Parameters ordered from most to least important:')
    print(np.fliplr((inputs[idx])))

def ann_surrogate_test():

    # number of output neurons
    n_out = samples.shape[1]

    # create a vanilla ANN surrogate
    surrogate = es.methods.ANN_Surrogate()

    # train the surrogate on the data
    n_iter = 20000

    # surrogate.train(theta, samples, n_iter, test_frac=test_frac, n_layers=2, n_neurons=100)
    # campaign.add_app(name='ann_campaign', surrogate=surrogate)
    # campaign.save_state()

    campaign = es.Campaign(load_state=True, file_path='ann_model_2804.pickle')
    surrogate = campaign.surrogate

    # evaluate the surrogate on the test data
    test_predictions = np.zeros([n_mc - I, n_out])
    for count, i in enumerate(range(I, n_mc)):
        theta_point = [x[i] for x in theta]
        test_predictions[count] = surrogate.predict(theta_point)

    # plot the test predictions and data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(test_predictions.T, 'b')
    ax.plot(samples[I:].T, 'r+')
    plt.tight_layout()
    plt.savefig('ann_test_data_res.png')

    # print the relative test error
    rel_err_test = np.linalg.norm(test_predictions - samples[I:]) / np.linalg.norm(test_predictions)
    print('Relative error on the test set is %.2f percent' % (rel_err_test * 100))

def gp_surroget_test():

    surrogate_gp = es.methods.GP_Surrogate(backend='mogp')

    # train the surrogate on the data
    samples_axial = samples[:, 0].reshape(-1, 1)

    n_out = samples_axial.shape[1]

    #surrogate_gp.train(theta, samples_axial, test_frac=test_frac)
    #campaign.add_app(name='gp_campaign', surrogate=surrogate_gp)
    #campaign.save_state()

    campaign = es.Campaign(load_state=True, file_path='mogp_model_2804.pickle')
    surrogate_gp = campaign.surrogate

    test_predictions = np.zeros([n_mc - I, n_out])
    test_pred_vars = np.zeros([n_mc - I, n_out])
    for count, i in enumerate(range(I, n_mc)):
        theta_point = [x[i] for x in theta]
        test_predictions[count], test_pred_vars[count] = surrogate_gp.predict(theta_point)

    # plot the test predictions and data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(samples_axial[I:], test_predictions, 'r+')
    plt.tight_layout()
    plt.savefig('gp_test_data_res.png')

    # print the relative test error
    rel_err_test = np.linalg.norm(test_predictions - samples_axial[I:]) / np.linalg.norm(samples_axial[I:])
    print('Relative error on the test set is %.2f percent' % (rel_err_test * 100))


theta = pd.read_pickle('inputs_2704.pickle').to_numpy()[:, :]
theta = theta.T.tolist()
theta = [np.array(x).reshape(-1, 1) for x in theta]
samples = pd.read_pickle('outputs_2704.pickle').to_numpy()[:, :]

n_mc = 500

# save the last 'test_frac' percent of the data for testing
test_frac = 0.25
I = np.int(len(samples) * (1.0 - test_frac))

campaign = es.Campaign()

# ===== ANN surrogate =====

#ann_surrogate_test()

# ===== GP surrogate =====

gp_surroget_test()
