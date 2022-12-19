import numpy as np
import csv
import pandas as pd

import easysurrogate as es


features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux', 'te_transp_flux_std', 'ti_transp_flux_std']

def load_csv_file(input_file='gem_data_625.txt', n_runs=625, input_dim=4, output_dim=2, std=False, startcol=2):

    input_samples = np.zeros((n_runs, input_dim))
    output_samples = np.zeros((n_runs, output_dim))

    with open(input_file, 'r') as inputfile:
        datareader = csv.reader(inputfile, delimiter=',')
        next(datareader) #DEBUG
        j_startcol = startcol #DEBUG
        i = 0
        for row in datareader:
            input_samples[i] = row[j_startcol:j_startcol+input_dim]
            output_samples[i] = row[j_startcol+input_dim:j_startcol+input_dim + output_dim]
            i = i + 1

    data = {}

    data['te_value'] = input_samples[:, 0].reshape(-1, 1)
    data['ti_value'] = input_samples[:, 1].reshape(-1, 1)
    data['te_ddrho'] = input_samples[:, 2].reshape(-1, 1)
    data['ti_ddrho'] = input_samples[:, 3].reshape(-1, 1)

    if not std and output_dim == 1:
        data['ti_transp_flux'] = output_samples[:, 1].reshape(-1, 1)

    if not std and output_dim == 2:
        data['te_transp_flux'] = output_samples[:, 0].reshape(-1, 1)
        data['ti_transp_flux'] = output_samples[:, 1].reshape(-1, 1)
    
    if std and output_dim == 2:
        data['ti_transp_flux'] = output_samples[:, 0].reshape(-1, 1)
        data['ti_transp_flux_std'] = output_samples[:, 1].reshape(-1, 1)
    
    if std and output_dim == 4:
        data['te_transp_flux'] = output_samples[:, 0].reshape(-1, 1)
        data['ti_transp_flux'] = output_samples[:, 1].reshape(-1, 1)
        data['te_transp_flux_std'] = output_samples[:, 2].reshape(-1, 1)
        data['ti_transp_flux_std'] = output_samples[:, 3].reshape(-1, 1)

    return data

# Get data to a hfd5

# Create an ES campaign
campaign = es.Campaign(load_state=False)

# Case from single flux tube GEM UQ campaign (4 parameters, tensor product of grid with 3 points per DoF) - longer runs, fixed a permutation of input-outputs
data = load_csv_file(input_file='resuq_main_ti_transp_flux_all_akgbbn1a_9.csv', 
                     n_runs=81,
                     output_dim=2,
                     std=True,
                     startcol=3,
                     )
campaign.store_data_to_hdf5(data, file_path='gem_uq_81_std.hdf5')

#TODO: Make script read the data based on the given UQ DB name given
