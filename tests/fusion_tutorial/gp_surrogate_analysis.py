import numpy as np
from matplotlib import pyplot as plt
import easysurrogate as es
from scipy.stats import wasserstein_distance as ws_dist


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

def main(order_num=None):

    campaign = es.Campaign(load_state=True, file_path='mogp_10_model_1006_full_100.pickle')

    # load the training and testing data from simulations
    data_frame_sim = campaign.load_hdf5_data(file_path='sim_data.hdf5')

    # load the predictions from surrogate
    data_frame_sur = campaign.load_hdf5_data(file_path='sur_data.hdf5')

    # defining name of input features
    feature_names = ["Qe_tot", "H0", "Hw", "chi", "Te_bc", "b_pos", "b_height", "b_sol", "b_width", "b_slope"]
    # choosing all features
    features = dict((k, data_frame_sim[k]) for k in feature_names)

    # preparing a list of feature to train
    theta = [v for k,v in features.items()]

    # chose target data from the loaded dictionary
    samples = data_frame_sim['Te']

    # chose predicted target from loaded dictionary
    predictions = data_frame_sur['Te']
    pred_vars = data_frame_sur['Te_sigmasq']

    # data size parameters
    ndim_in = campaign.surrogate.n_in
    ndim_out = campaign.surrogate.n_out
    n_mc_l = theta[0].shape[0]

    # getting order of input features, list of numbers corresponding to the feature name list
    if order_num is None:
        order_num = np.arange(ndim_in)

    # getting a subset of input data, take first ndim_in features defined by order_num list
    theta_reod = campaign.surrogate.feat_eng.chose_feat_subset(theta, ndim_in, order_num)


    # list of run indices used in training
    train_inds = campaign.surrogate.feat_eng.train_indices.tolist()
    test_inds = campaign.surrogate.feat_eng.test_indices.tolist()
    n_train = campaign.surrogate.feat_eng.n_train

    training_predictions = predictions[train_inds]
    training_pred_vars = pred_vars[train_inds]
    test_predictions = predictions[test_inds]
    test_pred_vars = pred_vars[test_inds]

    ############
    # Analysis #
    ############

    analysis = es.analysis.GP_analysis(campaign.surrogate)

    # plot the train predictions and original data
    analysis.plot_prediction_results(samples[train_inds].reshape(-1), training_predictions.reshape(-1),
                            training_pred_vars.reshape(-1), 1.0, 'gp_train_{}_data_res.png'.format(ndim_in))

    rel_err_train = np.linalg.norm(training_predictions - samples[train_inds]) / np.linalg.norm(samples[train_inds])
    print('Relative error on the training set is %.2f percent' % (rel_err_train * 100))

    # plot prediction against parameter values
    if len(theta_reod) == 1 and samples.shape[1] == 1:
        analysis.plot_prediction_results_vspar(theta_reod[0][train_inds].reshape(-1),
                                      samples[train_inds],
                                      training_predictions.reshape(-1),
                                      training_pred_vars.reshape(-1),
                                      name='gp_theta_train_{}_data_res.png'.format(ndim_in))


    # plot the test predictions and data
    analysis.plot_prediction_results(samples[test_inds].reshape(-1), test_predictions.reshape(-1),
                            test_pred_vars.reshape(-1), name='gp_test_{}_data_res.png'.format(ndim_in))

    # plot a several chosen test prediction as radial dependency
    analysis.plot_prediction_results_vectorqoi(samples[test_inds], test_predictions, test_pred_vars,
                                      name='gp_test_{}_data_res_profiles.png'.format(ndim_in))

    # plot prediction against parameter values for testing data
    if len(theta_reod) == 1 and samples.shape[1] == 1:
        analysis.plot_prediction_results_vspar(theta_reod[0][test_inds].reshape(-1),
                                      samples[test_inds],
                                      test_predictions.reshape(-1),
                                      test_pred_vars,
                                      name='gp_theta_test_{}_data_res.png'.format(ndim_in))

    # print the relative test error
    rel_err_test = np.linalg.norm(test_predictions - samples[test_inds]) / np.linalg.norm(samples[test_inds])
    print('Relative error on the test set is %.2f percent' % (rel_err_test * 100))

    # plot average predicted variance and R2 score on a test set
    test_pred_var_tot = test_predictions.var()
    print('Variance of predicted result means for the test set %.3f' % test_pred_var_tot)
    print('R2 score on testing set: {}'.format(campaign.surrogate.model.instance.score(
        np.array(theta_reod)[:, [test_inds]].reshape(n_mc_l - n_train, ndim_in), samples[test_inds])))

    ############
    # QoI PDFs #
    ############

    # analyse the QoI (Te(rho=0)) for test set
    te_ax_ts_dat_dom, te_ax_ts_dat_pdf = analysis.get_pdf(samples[test_inds][:, 0])
    te_ax_ts_surr_dom, te_ax_ts_surr_pdf = analysis.get_pdf(predictions[test_inds][:, 0])
    print('len of test data: {}'.format(samples[test_inds][:, 0].shape))

    te_ax_tr_dat_dom, te_ax_tr_dat_pdf = analysis.get_pdf(samples[train_inds][:, 0])
    te_ax_tr_surr_dom, te_ax_tr_surr_pdf = analysis.get_pdf(training_predictions[:, 0])
    print('len of train data: {}'.format(samples[train_inds][:, 0].shape))

    te_ax_tt_dat_dom, te_ax_tt_dat_pdf = analysis.get_pdf(samples[:][:, 0])
    tot_pred = np.concatenate([training_predictions, test_predictions])
    te_ax_tt_surr_dom, te_ax_tt_surr_pdf = analysis.get_pdf(tot_pred[:, 0])
    print('len of total data: {}'.format(samples[:][:, 0].shape))

    analysis.plot_pdfs(te_ax_ts_dat_dom, te_ax_ts_dat_pdf, te_ax_ts_surr_dom, te_ax_ts_surr_pdf,
                       names=['simulation_test', 'surrogate_test', 'simulation_train', 'surrogate_train'],
                       qoi_names=['Te(r=0)'], filename='pdf_qoi_trts_{}'.format(ndim_in))
    w_d = ws_dist(te_ax_ts_surr_pdf, te_ax_ts_dat_pdf)
    print('Wasserstein distance for distribution of selected QoI produced by simulation and surrogate: {}'.format(w_d))

    # plotting errors on a single-case basis, for axial value
    analysis.get_regression_error(np.array([theta_feat[test_inds].reshape(-1) for theta_feat in theta_reod]).T,
                                  samples[test_inds],
                                  np.array([theta_feat[train_inds].reshape(-1) for theta_feat in theta_reod]).T,
                                  samples[train_inds])

    return rel_err_test


if __name__ == '__main__':
    main()

