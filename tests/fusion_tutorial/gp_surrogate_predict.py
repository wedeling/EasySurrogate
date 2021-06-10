import numpy as np
import easysurrogate as es


def main(order_num=None):

    campaign = es.Campaign(load_state=True, file_path='mogp_10_model_1006_full_100.pickle')

    # load the training and testing data from simulations
    data_frame = campaign.load_hdf5_data(file_path='sim_data.hdf5')

    # defining name of input features
    feature_names = ["Qe_tot", "H0", "Hw", "chi", "Te_bc", "b_pos", "b_height", "b_sol", "b_width", "b_slope"]
    # choosing all features
    features = dict((k, data_frame[k]) for k in feature_names)

    # preparing a list of feature to train
    theta = [v for k,v in features.items()]

    ndim_in = campaign.surrogate.n_in
    ndim_out = campaign.surrogate.n_out
    n_mc_l = theta[0].shape[0]

    # getting order of input features, list of numbers corresponding to the feature name list
    if order_num is None:
        order_num = np.arange(ndim_in)

    # getting a subset of input data, take first ndim_in features defined by order_num list
    theta_reod = campaign.surrogate.feat_eng.chose_feat_subset(theta, ndim_in, order_num)


    ### Total dataset

    predictions = np.zeros([n_mc_l, ndim_out])
    pred_vars = np.zeros([n_mc_l, ndim_out])
    for i in range(n_mc_l):
        theta_point = [x[i] for x in theta_reod]
        predictions[i], pred_vars[i] = campaign.surrogate.predict(theta_point)

    data_sur = {}
    data_sur['Te'] = predictions
    data_sur['Te_sigmasq'] = pred_vars
    campaign.store_data_to_hdf5(data_sur, file_path='sur_data.hdf5')


if __name__ == "__main__":
    main()

