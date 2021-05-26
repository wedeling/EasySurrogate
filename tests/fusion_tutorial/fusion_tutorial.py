import easyvvuq as uq
import easysurrogate as es
import chaospy as cp
import time
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from scipy.stats import wasserstein_distance as ws_dist


def write_template(params):
    str = ""
    first = True
    for k in params.keys():
        if first:
            str += '{"%s": "$%s"' % (k, k)
            first = False
        else:
            str += ', "%s": "$%s"' % (k, k)
    str += '}'
    print(str, file=open('fusion.template', 'w'))


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
        "Qe_tot": cp.Uniform(1.8e6, 2.2e6),
        "H0": cp.Uniform(0.0, 0.2),
        "Hw": cp.Uniform(0.1, 0.5),
        "chi": cp.Uniform(0.8, 1.2),
        "Te_bc": cp.Uniform(80.0, 120.0),
        "a0": cp.Uniform(0.9, 1.1),
        "R0": cp.Uniform(2.7, 3.3),
        "E0": cp.Uniform(1.4, 1.6),
        "b_pos": cp.Uniform(0.95, 0.99),
        "b_height": cp.Uniform(5e19, 7e19),
        "b_sol": cp.Uniform(1e19, 3e19),
        "b_width": cp.Uniform(0.015, 0.025),
        "b_slope": cp.Uniform(0.005, 0.020)
    }
    vary_2 = {
        "Qe_tot": cp.Uniform(1.8e6, 2.2e6),
        "Te_bc": cp.Uniform(80.0, 120.0)
    }
    vary_5 = {
        "Qe_tot": cp.Uniform(1.8e6, 2.2e6),
        "H0": cp.Uniform(0.0, 0.2),
        "Hw": cp.Uniform(0.1, 0.5),
        "chi": cp.Uniform(0.8, 1.2),
        "Te_bc": cp.Uniform(80.0, 120.0)
    }
    vary_10 = {
        "Qe_tot": cp.Uniform(1.8e6, 2.2e6),
        "H0": cp.Uniform(0.0, 0.2),
        "Hw": cp.Uniform(0.1, 0.5),
        "chi": cp.Uniform(0.8, 1.2),
        "Te_bc": cp.Uniform(80.0, 120.0),
        "b_pos": cp.Uniform(0.95, 0.99),
        "b_height": cp.Uniform(5e19, 7e19),
        "b_sol": cp.Uniform(1e19, 3e19),
        "b_width": cp.Uniform(0.015, 0.025),
        "b_slope": cp.Uniform(0.005, 0.020)
    }
    return vary_10


def get_inputs(campaign, sampler, n_mc):
    # number of inputs
    D = len(sampler.vary.get_keys())
    # store the parameters in theta, will be used as features
    theta = np.zeros([n_mc, D])
    # loop over all runs
    for i, run in enumerate(campaign.list_runs()):
        # get the parameter values
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


def get_sa_order(feat_names_orig, feat_names_sa):
    """
    Creates a relative permutation of feature indices
    Args:
        feat_names_orig: list of features names in the original order, as it appears in easyvvuq vary
        feat_names_sa: list of features names in the order of their contribution to variance, got from SA

    Returns: list of numbers that define indices of the latter list in terms of the first one
    """

    order = []

    for f in feat_names_orig:
        order.append(feat_names_sa.index(f))

    return order


def chose_feat_subset(theta, res_dim_num=1, order=None):
    """
    Chooses subset of features
    Args:
        theta: a list of feature arrays
        res_dim_num: number of dimension ot include in the new feature list
        order: list of feature indices defining order in which features will be chosen

    Returns: list of feature arrays of length res_dim_num
    """

    if order is None:
        order = np.arange(len(theta))

    res_order = order[:res_dim_num]

    theta = [theta[i] for i in res_order]

    return theta


def plot_prediction_results(original_values, predicted_values,
                            predicted_variance=None, sigma_prefactor=1.0, name='gp_res.png'):
    """
    Plots scalar value from original and predicted value around y=x line
    Args:
        original_values:
        predicted_values:
        predicted_variance:
        sigma_prefactor:
        name:

    Returns:

    """

    # prefactor to determine the percentiles around mean of normal distribution
    # 0.13 is ~5% percentile from mean for gaussian,
    # 0.02 is ~1%,
    # 1.96 is 95%
    # 2.0  is 96%

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(original_values, original_values, 'ko', markersize=5, label='original data points')

    if predicted_variance is not None:
        ax.errorbar(original_values, predicted_values,
                    sigma_prefactor * predicted_variance, fmt='+', label='predictions')
    else:
        ax.plot(original_values, predicted_values, 'r+')

    plt.tight_layout()
    plt.savefig(name)


def plot_prediction_results_vspar(orig_x, orig_y, pred_y, pred_std, sigma_prefactor=1.0, name='gp_res_r.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if orig_y.shape[1] != 1:
        orig_y = orig_y[:, 0]
    ax.plot(orig_x, orig_y, 'r+', label='simulation')
    ax.errorbar(orig_x, pred_y, sigma_prefactor * pred_std,
                fmt='b+', label='surrogate')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig(name)


def plot_prediction_results_vectorqoi(orig_y, pred_y, pred_var=None,
                                      sigma_prefactor=1.0, freq=25, name='gp_res_vec_qoi.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x_rho = np.arange(100)

    for i, j in enumerate(range(pred_y.shape[0] // freq)):

        ax.plot(x_rho, orig_y[i].T, 'r-')

        if pred_var is not None:
            ax.errorbar(x=x_rho, y=pred_y[j, :].reshape(-1),
                        yerr=sigma_prefactor * pred_var[j, :].reshape(-1), fmt='b+')
        else:
            ax.plot(pred_y[j].T, 'b')

    plt.tight_layout()
    plt.savefig(name)


def plot_er_time(errors, times):
    """
    Plots time to train surrogate and surrogate testing RMSE error against number of input feature dimension
    Args:
        errors: RMSE of surrogate on testing dataset
        times: list of times spent on surrogate training in seconds

    Returns:

    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(2, 2 + len(errors))

    ax.plot(x, errors, color='b', label='rel. err.')
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_ylabel('%')
    ax.set_xlabel('n dim')

    ax2 = ax.twinx()
    ax2.plot(x, times, color='r', label='time to train')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylabel('s')

    plt.legend(loc='best')

    plt.title('test error and time to train models against number of input features')
    plt.savefig('err_and_time.png')
    plt.close()


def ann_derivative_based_sa(surrogate, keys, D=10):
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
        # use recursive formulas to update the mean (and variance although I don't use it)
        # with a new f_grad_x2 sample
        mean_f_grad_x2, var2 = analysis.recursive_moments(f_grad_x2, mean_f_grad_x2, var2, i)

    print('(E_dx_f(x))^2 : {}'.format(mean_f_grad_x2))
    # get all input names
    inputs = np.array(list(keys))
    # sort the sensitivity indices
    idx = np.argsort(np.abs(mean_f_grad_x2).T)
    # print input in order of importance
    print('Parameters ordered from most to least important:')
    order = np.fliplr((inputs[idx]))
    print(order)

    return order


def gp_derivative_based_sa(surrogate, X, keys, D=10):
    # get the derivatives
    f_grad_x = surrogate.derivative_x([x[0] for x in X])
    # square it
    mean_f_grad_x2 = f_grad_x ** 2
    var2 = np.zeros(D)
    # create a basic EasySurrogate analysis class
    analysis = es.analysis.BaseAnalysis()

    # loop over all training samples
    for i in range(1, X[0].shape[0]):
        x_point = [x[i] for x in X]
        f_grad_x2 = surrogate.derivative_x(x_point) ** 2
        # with a new f_grad_x2 sample
        mean_f_grad_x2, var2 = analysis.recursive_moments(f_grad_x2, mean_f_grad_x2, var2, i)

    print('(E_dx_f(x))^2 : {}'.format(mean_f_grad_x2))
    # get all input names
    inputs = np.array(list(keys))
    # sort the sensitivity indices
    idx = np.argsort(np.abs(mean_f_grad_x2.T).T)
    # print input in order of importance
    print('Parameters ordered from most to least important:')
    order = np.fliplr((inputs[idx]))
    print(order)

    return order


def ann_surrogate_test(samples_c, n_train):
    campaign = es.Campaign()

    # create a vanilla ANN surrogate
    surrogate_ann = es.methods.ANN_Surrogate()

    # -- If not using all the grid points
    # samples_c = samples_c[:, 0].reshape(-1, 1)  # axial
    # samples_c = samples_c[:, np.arange(0, 100, 5).tolist()]  # sparse

    # number of output neurons
    n_out = samples_c.shape[1]

    # train the surrogate on the data
    n_iter = 20000

    surrogate_ann.train(theta, samples_c, n_iter, test_frac=test_frac, n_layers=2, n_neurons=1)
    campaign.add_app(name='ann_campaign', surrogate=surrogate_ann)
    campaign.save_state()

    # -- If loading a pretrained surrogate
    # campaign = es.Campaign(load_state=True)
    # surrogate = campaign.surrogate

    # evaluate the surrogate on the test data
    test_predictions = np.zeros([n_mc - n_train, n_out])
    for count, i in enumerate(range(n_train, n_mc)):
        theta_point = [x[i] for x in theta]
        test_predictions[count] = surrogate_ann.predict(theta_point)

    # plot the test predictions and data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(test_predictions.T, 'b')
    ax.plot(samples[n_train:].T, 'r+')
    plt.tight_layout()
    plt.savefig('ann_ax_test_data_res.png')

    # -- If plotting around y=x for a scalar QoI
    # plot_prediction_results(samples_c[I:], test_predictions)

    # print the relative test error
    rel_err_test = np.linalg.norm(test_predictions - samples_c[n_train:]) / np.linalg.norm(test_predictions)
    print('Relative error on the test set is %.2f percent' % (rel_err_test * 100))

    # perfrom derivative based surrogate
    ann_derivative_based_sa(surrogate_ann, order_orig)


def gp_surrogate_test(order=None, ndim=None, n_train=None):
    ### TRAINING PHASE
    campaign = es.Campaign()

    surrogate_gp = es.methods.GP_Surrogate(backend='scikit-learn')

    # -- If chosing data not on every grid point
    # samples_c= samples_c[:, 0].reshape(-1, 1)
    # samples_c = samples_c[:, np.arange(0, 100, 20).tolist()]

    n_out = samples_c.shape[1]
    n_mc_l = samples_c.shape[0]

    if ndim is None:
        if isinstance(theta_c, list):
            ndim = theta_c[0].shape[0]
        elif isinstance(theta_c, np.ndarray):
            ndim = theta_c.shape[1]

    if order is None:
        order = np.arange(ndim)

    theta_reod = chose_feat_subset(theta_c, ndim, order)

    day = date.today().strftime('%d%m')
    st_time = time.time()
    surrogate_gp.train(theta_reod, samples_c, test_frac=test_frac, basekernel='Matern', noize='fit')
    tot_time = time.time() - st_time
    print('Time to train a GP surrogate {:.3}'.format(tot_time))

    campaign.add_app(name='gp_campaign', surrogate=surrogate_gp)
    campaign.save_state(file_path='mogp_{}_model_{}_full_100.pickle'.format(ndim, day))

    # -- If loading a pretrained surrogate
    # campaign = es.Campaign(load_state=True, file_path='skl_1_model_2505_full_100.pickle')
    # surrogate_gp = campaign.surrogate

    ##### TRYING SEQUENTIAL OPTIMISATION - passes, but resulting model worsesns in its performance
    # surrogate_gp.train_sequentially(theta_reod, samples_axial, n_iter=10, acquisition_function='poi')
    # print('Indices of runs used for training: {}'.format(surrogate_gp.feat_eng.train_indices))
    #####

    surrogate_gp.model.print_model_info()

    # list of run indices used in training
    train_inds = surrogate_gp.feat_eng.train_indices.tolist()
    test_inds = surrogate_gp.feat_eng.test_indices.tolist()
    n_train = surrogate_gp.feat_eng.n_train  # length of training data set

    ### ANALYSIS PHASE
    analysis = es.analysis.GP_analysis(surrogate_gp)

    ### Training set
    # evaluate the surrogate on the training data
    training_predictions = np.zeros([n_train, n_out])
    training_pred_vars = np.zeros([n_train, n_out])

    for i, j in enumerate(train_inds):
        theta_point = [x[j] for x in theta_reod]
        training_predictions[i], training_pred_vars[i] = surrogate_gp.predict(theta_point)

    # plot the train predictions and original data
    plot_prediction_results(samples_c[train_inds].reshape(-1), training_predictions.reshape(-1),
                            training_pred_vars.reshape(-1), 1.0, 'gp_train_{}_data_res.png'.format(ndim))

    rel_err_train = np.linalg.norm(training_predictions - samples_c[train_inds]) / np.linalg.norm(samples_c[train_inds])
    print('Relative error on the training set is %.2f percent' % (rel_err_train * 100))

    # plot prediction against parameter values
    if len(theta_reod) == 1 and samples_c.shape[1] == 1:
        plot_prediction_results_vspar(theta_reod[0][train_inds].reshape(-1),
                                      samples_c[train_inds],
                                      training_predictions.reshape(-1),
                                      training_pred_vars.reshape(-1),
                                      name='gp_theta_train_{}_data_res.png'.format(ndim))

    ## Testing set
    # evaluate on testing data
    test_predictions = np.zeros([n_mc_l - n_train, n_out])
    test_pred_vars = np.zeros([n_mc_l - n_train, n_out])
    for count, i in enumerate(range(n_train, n_mc_l)):
        theta_point = [x[i] for x in theta_reod]
        test_predictions[count], test_pred_vars[count] = surrogate_gp.predict(theta_point)

    # plot the test predictions and data
    plot_prediction_results(samples_c[test_inds].reshape(-1), test_predictions.reshape(-1),
                            test_pred_vars.reshape(-1), name='gp_test_{}_data_res.png'.format(ndim))

    # plot a several chosen test prediction as radial dependency
    plot_prediction_results_vectorqoi(samples_c[test_inds], test_predictions, test_pred_vars,
                                      name='gp_test_{}_data_res_profiles.png'.format(ndim))

    # plot prediction against parameter values for testing data
    if len(theta_reod) == 1 and samples_c.shape[1] == 1:
        plot_prediction_results_vspar(theta_reod[0][test_inds].reshape(-1),
                                      samples_c[test_inds],
                                      test_predictions.reshape(-1),
                                      test_pred_vars,
                                      name='gp_theta_test_{}_data_res.png'.format(ndim))

    # print the relative test error
    rel_err_test = np.linalg.norm(test_predictions - samples_c[test_inds]) / np.linalg.norm(samples_c[test_inds])
    print('Relative error on the test set is %.2f percent' % (rel_err_test * 100))

    # plot average predicted variance and R2 score on a test set
    test_pred_var_tot = test_predictions.var()
    print('Variance of predicted result means for the test set %.3f' % test_pred_var_tot)
    print('R2 score on testing set: {}'.format(surrogate_gp.model.instance.score(
        np.array(theta_reod)[:, [test_inds]].reshape(n_mc - n_train, ndim), samples_c[test_inds])))

    ### SENSITIVITY ANALYSIS
    if surrogate_gp.backend == 'mogp':
        gp_derivative_based_sa(surrogate_gp, theta_reod[:][test_inds], keys=order_orig)

    ### QoI pdfs
    # analyse the QoI (Te(rho=0)) for test set
    te_ax_ts_dat_dom, te_ax_ts_dat_pdf = analysis.get_pdf(samples_c[test_inds][:, 0])
    te_ax_ts_surr_dom, te_ax_ts_surr_pdf = analysis.get_pdf(test_predictions[:, 0])
    print('len of test data: {}'.format(samples_c[test_inds][:, 0].shape))

    te_ax_tr_dat_dom, te_ax_tr_dat_pdf = analysis.get_pdf(samples_c[train_inds][:, 0])
    te_ax_tr_surr_dom, te_ax_tr_surr_pdf = analysis.get_pdf(training_predictions[:, 0])
    print('len of train data: {}'.format(samples_c[train_inds][:, 0].shape))

    te_ax_tt_dat_dom, te_ax_tt_dat_pdf = analysis.get_pdf(samples_c[:][:, 0])
    tot_pred = np.concatenate([training_predictions, test_predictions])
    te_ax_tt_surr_dom, te_ax_tt_surr_pdf = analysis.get_pdf(tot_pred[:, 0])
    print('len of total data: {}'.format(samples_c[:][:, 0].shape))

    analysis.plot_pdfs(te_ax_ts_dat_dom, te_ax_ts_dat_pdf, te_ax_ts_surr_dom, te_ax_ts_surr_pdf,
                       names=['simulation_test', 'surrogate_test', 'simulation_train', 'surrogate_train'],
                       qoi_names=['Te(r=0)'], filename='pdf_qoi_trts_{}'.format(ndim))
    w_d = ws_dist(te_ax_ts_surr_pdf, te_ax_ts_dat_pdf)
    print('Wasserstein distance for distribution of selected QoI produced by simulation and surrogate: {}'.format(w_d))

    # plotting errors on a single-case basis, for axial value
    analysis.get_regression_error(np.array([theta_feat[test_inds].reshape(-1) for theta_feat in theta_reod]).T,
                                  samples_c[test_inds],
                                  np.array([theta_feat[train_inds].reshape(-1) for theta_feat in theta_reod]).T,
                                  samples_c[train_inds])

    return tot_time, rel_err_test


# === Analysing surrogates for Fusion tutorial code ===

# reading input and output data of simulations
theta = pd.read_pickle('inputs_2704.pickle').to_numpy()[:, :]
theta = theta.T.tolist()
theta = [np.array(x).reshape(-1, 1) for x in theta]

samples = pd.read_pickle('outputs_2704.pickle').to_numpy()[:, :]

# Choosing the X, Y subsets to analyse
n_mc = 500
samples_c = samples[:n_mc]
theta_c = [t[:n_mc] for t in theta]

# save the last 'test_frac' percent of the data for testing
test_frac = 0.8
n_train = int((1. - test_frac) * n_mc)

# Defining the current ordering of parameters by sensitivity
order_orig = ["Qe_tot", "H0", "Hw", "chi", "Te_bc", "b_pos", "b_height", "b_sol", "b_width", "b_slope"]

# -- If we change order of the features by their importance
# order = get_sa_order(order_orig, order_sa_ann)

order = np.arange(len(order_orig))

# ===== ANN surrogate =====

# ann_surrogate_test(samples_c, n_train)

# ===== GP surrogate =====
times = []
errors = []

for i in range(1, 11):
    t, e = gp_surrogate_test(order, i)
    times.append(t)
    errors.append(e)

plot_er_time(errors, times)
