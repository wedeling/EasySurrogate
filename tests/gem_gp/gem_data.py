import numpy as np
import csv

import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux']

def load_csv_file():
    N_runs = 625
    frac_train = 0.5
    input_dim = 4
    output_dim = 2

    input_samples = np.zeros((N_runs, input_dim))
    output_samples = np.zeros((N_runs, output_dim))

    input_file = 'gem_data_625.txt'

    with open(input_file, 'r') as inputfile:
        datareader = csv.reader(inputfile, delimiter=',')
        i = 0
        for row in datareader:
            input_samples[i] = row[0:input_dim]
            output_samples[i] = row[input_dim:input_dim+output_dim]
            i = i + 1

    data = {}
    data['te_value'] = input_samples[:, 0].reshape(-1, 1)
    data['ti_value'] = input_samples[:, 1].reshape(-1, 1)
    data['te_ddrho'] = input_samples[:, 2].reshape(-1, 1)
    data['ti_ddrho'] = input_samples[:, 3].reshape(-1, 1)
    data['te_transp_flux'] = output_samples[:, 0].reshape(-1, 1)
    data['ti_transp_flux'] = output_samples[:, 1].reshape(-1, 1)

    return data

# get dat ato a hfd5
campaign = es.Campaign(load_state=False)
data = load_csv_file()
campaign.store_data_to_hdf5(data, file_path='gem_data_625.hdf5')
# TODO try on a larger dataset from a MFW
