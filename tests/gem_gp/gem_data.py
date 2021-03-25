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

def load_csv_dict_file(input_file='gem0_lhc_res.csv', n_runs=1000):
    Xlabels = ['te.value', 'ti.value', 'te.ddrho', 'ti.ddrho']
    Ylabels = ['te.flux', 'ti.flux']

    input_dim = 4
    output_dim = 2

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
    data['te_value'] = input_samples[:, 0].reshape(-1, 1)
    data['ti_value'] = input_samples[:, 1].reshape(-1, 1)
    data['te_ddrho'] = input_samples[:, 2].reshape(-1, 1)
    data['ti_ddrho'] = input_samples[:, 3].reshape(-1, 1)
    data['te_transp_flux'] = output_samples[:, 0].reshape(-1, 1)
    data['ti_transp_flux'] = output_samples[:, 1].reshape(-1, 1)

    return data

def load_wf_csv_file(data_dir='', input_file='AUG_gem_inoutput.txt'):

    Xlabels = ['Te-ft5', 'Ti-ft5', 'dTe-ft5', 'dTi-ft5']
    Ylabels = ['flux-Te-ft5', 'flux-Ti-ft5']

    input_samples = []
    output_samples = []

    with open(data_dir + input_file, 'r') as inputfile:
        datareader = csv.reader(inputfile, delimiter=' ') # Could be done with DictReader(), then no need in column numbers
        column_names = next(datareader)
        x_column_numbers = [column_names.index(x) for x in Xlabels]
        y_column_numbers = [column_names.index(y) for y in Ylabels]
        for row in datareader:
            input_samples.append([float(row[i]) for i in x_column_numbers])
            output_samples.append([float(row[i]) for i in y_column_numbers])

    input_samples = np.array(input_samples)
    output_samples = np.array(output_samples)

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

#data = load_wf_csv_file(input_file='AUG_gem_inoutput.txt')
#campaign.store_data_to_hdf5(data, file_path='gem_workflow.hdf5')

#data = load_csv_dict_file()
#campaign.store_data_to_hdf5(data, file_path='gem0_lhc.hdf5')
