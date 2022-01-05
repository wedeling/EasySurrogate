import pandas as pd
import numpy as np
import chaospy as cp
import easysurrogate as es


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


def main():

    campaign = es.Campaign()
    feat_eng = es.methods.Feature_Engineering()

    # reading input and output data of simulations
    theta = pd.read_pickle('inputs_2704.pickle').to_numpy()[:, :]
    theta = theta.T.tolist()
    theta = [np.array(x).reshape(-1, 1) for x in theta]

    samples = pd.read_pickle('outputs_2704.pickle').to_numpy()[:, :]

    # Choosing the X, Y subsets to analyse
    ndim_in = 10
    n_mc = 500
    samples_c = samples[:n_mc]
    theta_c = [t[:n_mc] for t in theta]

    # Defining the current ordering of parameters by sensitivity
    name_order_orig = [
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

    # -- If we change order of the features by their importance
    # order = get_sa_order(order_orig, order_sa_ann)

    order_num = np.arange(len(name_order_orig))

    theta_reod = feat_eng.chose_feat_subset(theta_c, ndim_in, order_num)

    # save reference data as hdf

    data_sim = {}
    data_sim['Te'] = samples_c
    for i, name in enumerate(name_order_orig):
        data_sim[name] = theta_reod[i]
    campaign.store_data_to_hdf5(data_sim, file_path='sim_data.hdf5')


if __name__ == "__main__":
    main()
