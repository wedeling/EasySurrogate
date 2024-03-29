import numpy as np
import csv
import pandas as pd

import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux']


def load_csv_file(input_file='gem_data_625.txt', n_runs=625):
    input_dim = 4
    output_dim = 2

    input_samples = np.zeros((n_runs, input_dim))
    output_samples = np.zeros((n_runs, output_dim))

    with open(input_file, 'r') as inputfile:
        datareader = csv.reader(inputfile, delimiter=',')
        i = 0
        for row in datareader:
            input_samples[i] = row[0:input_dim]
            output_samples[i] = row[input_dim:input_dim + output_dim]
            i = i + 1

    data = {}
    data['te_value'] = input_samples[:, 0].reshape(-1, 1)
    data['ti_value'] = input_samples[:, 1].reshape(-1, 1)
    data['te_ddrho'] = input_samples[:, 2].reshape(-1, 1)
    data['ti_ddrho'] = input_samples[:, 3].reshape(-1, 1)
    data['te_transp_flux'] = output_samples[:, 0].reshape(-1, 1)
    data['ti_transp_flux'] = output_samples[:, 1].reshape(-1, 1)

    return data


def load_csv_dict_file(input_file='gem0_lhc_res.csv', n_runs=1000, input_dim=4, output_dim=2):
    if input_dim == 4:
        Xlabels = ['te.value', 'ti.value', 'te.ddrho', 'ti.ddrho']
    elif input_dim == 2:
        Xlabels = ['te.ddrho', 'ti.ddrho']
    if output_dim == 2:
        Ylabels = ['te.flux', 'ti.flux']
    if output_dim == 1:
        Ylabels = ['ti.flux']

    input_samples = []
    output_samples = []

    with open(input_file, 'r') as inputfile:
        datareader = csv.DictReader(inputfile, delimiter=',')
        for row in datareader:
            input_samples.append([float(row[x]) for x in Xlabels])
            output_samples.append([float(row[y]) for y in Ylabels])

    input_samples = np.array(input_samples)
    output_samples = np.array(output_samples)

    data = {}

    if input_dim == 4:
        data['te_value'] = input_samples[:, 0].reshape(-1, 1)
        data['ti_value'] = input_samples[:, 1].reshape(-1, 1)
        data['te_ddrho'] = input_samples[:, 2].reshape(-1, 1)
        data['ti_ddrho'] = input_samples[:, 3].reshape(-1, 1)
    elif input_dim == 2:
        data['te_ddrho'] = input_samples[:, 0].reshape(-1, 1)
        data['ti_ddrho'] = input_samples[:, 1].reshape(-1, 1)
    if output_dim == 2:
        data['te_transp_flux'] = output_samples[:, 0].reshape(-1, 1)
        data['ti_transp_flux'] = output_samples[:, 1].reshape(-1, 1)
    elif output_dim == 1:
        data['ti_transp_flux'] = output_samples[:, 0].reshape(-1, 1)

    return data


def load_wf_csv_file(data_dir='', input_file='AUG_gem_inoutput.txt', runs=[0, 500]):

    Xlabels = ['Te-ft5', 'Ti-ft5', 'dTe-ft5', 'dTi-ft5']
    Ylabels = ['flux-Te-ft5', 'flux-Ti-ft5']

    input_samples = []
    output_samples = []

    with open(data_dir + input_file, 'r') as inputfile:
        # Could be done with DictReader(), then no need in column numbers
        datareader = csv.reader(inputfile, delimiter=' ')
        column_names = next(datareader)
        x_column_numbers = [column_names.index(x) for x in Xlabels]
        y_column_numbers = [column_names.index(y) for y in Ylabels]
        for row in datareader:
            input_samples.append([float(row[i]) for i in x_column_numbers])
            output_samples.append([float(row[i]) for i in y_column_numbers])

    input_samples = np.array(input_samples)
    output_samples = np.array(output_samples)

    run_first = runs[0]
    run_last = runs[-1]

    data = {}
    data['te_value'] = input_samples[run_first:run_last, 0].reshape(-1, 1)
    data['ti_value'] = input_samples[run_first:run_last, 1].reshape(-1, 1)
    data['te_ddrho'] = input_samples[run_first:run_last, 2].reshape(-1, 1)
    data['ti_ddrho'] = input_samples[run_first:run_last, 3].reshape(-1, 1)
    data['te_transp_flux'] = output_samples[run_first:run_last, 0].reshape(-1, 1)
    data['ti_transp_flux'] = output_samples[run_first:run_last, 1].reshape(-1, 1)

    return data

# get data to a hfd5


# create an ES campaign
campaign = es.Campaign(load_state=False)

# 1) case for data from single flux tube GEM UQ campaign
#data = load_csv_file()
#campaign.store_data_to_hdf5(data, file_path='gem_data_625.hdf5')

# 2) case for data from a MFW production run
#data = load_wf_csv_file(input_file='AUG_gem_inoutput.txt')
#campaign.store_data_to_hdf5(data, file_path='gem_workflow_500.hdf5')

# 3) case for data generated from single flux tube GEM0 with 4 parameters (LHD, with a wrapper)
#data = load_csv_dict_file()
#campaign.store_data_to_hdf5(data, file_path='gem0_lhc.hdf5')

# 4) case for from single flux tube GEM0 with 2 parameters (LHD, with a wrapper)
data = load_csv_dict_file(input_file='gem0_lhc_256.csv', n_runs=256, input_dim=2, output_dim=1)
campaign.store_data_to_hdf5(data, file_path='gem0_lhc_256.hdf5')
