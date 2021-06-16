import numpy as np
import easysurrogate as es
import time
from datetime import date


def gp_train(order_num=None, ndim_in=None, ndim_out=None, n_train=None, test_frac=0.0):

    # creat e campaign object
    campaign = es.Campaign()

    # load th training and testing data from simulations
    data_frame = campaign.load_hdf5_data()

    # chose target data from the loaded dictionary
    target = data_frame['Te']

    # -- If chosing data not on every grid point
    # samples = samples_c[:, 0].reshape(-1, 1)
    # samples = samples_c[:, np.arange(0, 100, 20).tolist()]

    # defining name of input features
    feature_names = [
        "Qe_tot",
        "H0",
        "Hw",
        "chi",
        "Te_bc",
        "b_pos",
        "b_height",
        "b_sol",
        "b_width",
        "b_slope"]
    # choosing all features
    features = dict((k, data_frame[k]) for k in feature_names)

    # preparing a list of feature to train
    features = [v for k, v in features.items()]

    # number of samples
    n_mc_l = target.shape[0]

    # defining input dimensionality
    if ndim_in is None:
        if isinstance(features, list):
            #ndim_in = theta[0].shape[0]
            ndim_in = len(features)
        elif isinstance(features, np.ndarray):
            ndim_in = features.shape[1]
        else:
            ndim_in = 1
    else:
        ndim_in = len(features)

    # defining output dimensionality
    if ndim_out is None:
        ndim_out = target.shape[1]

    # getting order of input features, list of numbers corresponding to the feature name list
    if order_num is None:
        order_num = np.arange(ndim_in)

    # create a surrogate object
    surrogate = es.methods.GP_Surrogate(n_in=ndim_in, n_out=ndim_out, backend='scikit-learn')

    # getting a subset of input data, take first ndim_in features defined by order_num list
    features = surrogate.feat_eng.chose_feat_subset(features, ndim_in, order_num)

    # train a surrogate model
    day = date.today().strftime('%d%m')
    st_time = time.time()
    surrogate.train(features, target, test_frac=test_frac, basekernel='Matern', noize='fit')
    tot_time = time.time() - st_time
    print('Time to train a GP surrogate {:.3}'.format(tot_time))

    # print parameters of resulting surrogate
    surrogate.model.print_model_info()

    # save the app and the surrogate
    campaign.add_app(name='gp_campaign', surrogate=surrogate)
    campaign.save_state(file_path='mogp_{}_model_{}_full_100.pickle'.format(ndim_in, day))

    return tot_time


def main(**kwargs):

    test_frac = kwargs['test_frac']

    t = gp_train(test_frac=test_frac)

    print('Time to train: {} s'.format(t))


if __name__ == "__main__":
    main(test_frac=0.8)
