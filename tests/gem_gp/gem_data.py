import numpy as np
import csv
import pandas as pd

import easysurrogate as es


features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux', 'te_transp_flux_std', 'ti_transp_flux_std']

def load_csv_to_dict(input_file='gem_data_625.txt', n_runs=625, input_dim=4, output_dim=2, std=False):
    """
    Loads a CSV file with GEM/GEM0 data contatining columns for input and ouputs into a dictionary
    """

    data = pd.read_csv(input_file, sep=',')
    data = data[[*features_names, *target_names[:output_dim], 'ft']]
    data = data.to_dict(orient='list')
    data = {k:np.array(v).reshape(-1,1) for k,v in data.items()}

    data['ft'] = data['ft'].reshape(-1)

    return data

def load_csv_file(input_file='gem_data_625.txt', n_runs=625, input_dim=4, output_dim=2, std=False, startcol=2):

    input_samples = np.zeros((n_runs, input_dim))
    output_samples = np.zeros((n_runs, output_dim*2)) # also add stem and acn to targets

    with open(input_file, 'r') as inputfile:
        datareader = csv.reader(inputfile, delimiter=',')
        next(datareader) #skip header - ...
        j_startcol = startcol # redundant variable #DEBUG
        #i = 0
        for i,row in enumerate(datareader):
            row = np.array(row)
            row[row=='']='0.0'
            input_samples[i] = row[j_startcol:j_startcol+input_dim]
            output_samples[i] = row[j_startcol+input_dim:j_startcol+input_dim + output_dim*2]
            #i = i + 1

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
        order = [0,3,1,4] # precise ordering can change with da_utils.py:produce_stats_dataframes
        data['te_transp_flux'] = output_samples[:, order[0]].reshape(-1, 1)
        data['ti_transp_flux'] = output_samples[:, order[1]].reshape(-1, 1)
        data['te_transp_flux_std'] = output_samples[:, order[2]].reshape(-1, 1)
        data['ti_transp_flux_std'] = output_samples[:, order[3]].reshape(-1, 1)

    return data

def split_flux_tubes(data_dict, ft_len):
    """
    Split resulting dictionary into multiple dictionaries, one for each flux tube.
    Data separation between different flux tubes can be defined by:
        - fixed number of runs per flux tube
        - array of first runs for each flux tube
    """

    n_tot = data['ti_value'].size # unhardcode key
    n_ft = n_tot // ft_len

    print(data_dict)###DEBUG

    # Option 1: make a dictionary with keys being differeny flux tube strings and values being dictionaries 
    #    - current storing function does not support tree-like dictionaries
    data_dict_ft = {}    
    for i in range(n_ft):
        #data_dict_ft['ft'+str(i+1)] = {k:np.array(v[i*ft_len:(i+1)*ft_len]) for (k,v) in data_dict.items()}
        mask = [data_dict['ft']==i][0][:][:]
        print(mask) ###DEBUG
        data_dict_ft['ft'+str(i+1)] = {k:v[mask, :] for (k,v) in data_dict.items() if k!='ft'}

    # Option 2: 
    #   a. multiple files for flux tubes
    #   b. multiple quantity names
    #   c. each field with an array for different location, pad with default values
    #   d. each field with an array for different location, pad with None
    data_dict_list = []
    for i in range(n_ft):
        mask = [data_dict['ft']==i][0][:][:]
        #data_dict_list.append({k:np.array(v[i*ft_len:(i+1)*ft_len]) for (k,v) in data_dict.items()}) #assume that entries are ordered by flux tube number and there is the same number of rows per flux tube
        data_dict_list.append({k:np.array(v[mask]) for (k,v) in data_dict.items() if k!='ft'})

    #print(f"dimensions of original arrays: {data_dict['ti_transp_flux_std'].shape} ; and new arrays: {data_dict_list[0]['ti_transp_flux_std'].shape}") ###DEBUG

    return data_dict_list

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

# Get data to a hfd5

# create an ES campaign
campaign = es.Campaign(load_state=False)

# 1) Case for data from single flux tube GEM UQ campaign
#data = load_csv_file()
#campaign.store_data_to_hdf5(data, file_path='gem_data_625.hdf5')

# 2) Case for data from a MFW production run
#data = load_wf_csv_file(input_file='AUG_gem_inoutput.txt')
#campaign.store_data_to_hdf5(data, file_path='gem_workflow_500.hdf5')

# 3) Case for data generated from single flux tube GEM0 with 4 parameters (LHD, with a wrapper)
#data = load_csv_dict_file()
#campaign.store_data_to_hdf5(data, file_path='gem0_lhc.hdf5')

# 4) Case for from single flux tube GEM0 with 2 parameters (LHD, with a wrapper)
#data = load_csv_dict_file(input_file='gem0_lhc_256.csv', n_runs=256, input_dim=2, output_dim=1)
#campaign.store_data_to_hdf5(data, file_path='gem0_lhc_256.hdf5')

# 5) Case from single flux tube GEM UQ campaign (4 parameters, tensor product of grid with 2 points per DoF)
#data = load_csv_file(input_file='resuq_main_ti_transp_flux_all_moj202gj_11.csv', 
#                     n_runs=16,
#                     #input_dim=4, 
#                     output_dim=2,
#                     std=True
#                     )
#campaign.store_data_to_hdf5(data, file_path='gem_uq_16_std.hdf5')

# 5') Case from single flux tube GEM UQ campaign (4 parameters, tensor product of grid with 3 points per DoF)
# data = load_csv_file(input_file='resuq_main_ti_transp_flux_all_akgbbn1a_5.csv', 
#                      n_runs=79,
#                      #input_dim=4, 
#                      output_dim=2,
#                      std=True,
#                      startcol=3,
#                      )
# campaign.store_data_to_hdf5(data, file_path='gem_uq_79_std.hdf5')

# 5'') Case from single flux tube GEM UQ campaign (4 parameters, tensor product of grid with 3 points per DoF) - longer runs, fixed a permutation of input-outputs
# data = load_csv_file(input_file='resuq_main_ti_transp_flux_all_akgbbn1a_9.csv', 
#                      n_runs=81,
#                      #input_dim=4, 
#                      output_dim=4, #2,
#                      std=True,
#                      startcol=3,
#                      )
# campaign.store_data_to_hdf5(data, file_path='gem_uq_81_full.hdf5')

# 6) Cases predicted by AL GPR model, the values should yield results close to 2099023.289881937 
# data = load_csv_file(input_file='resuq_main_ti_transp_flux_all_alcand_30112022.csv', 
#                      n_runs=6,
#                      #input_dim=4, 
#                      output_dim=2,
#                      std=True,
#                      startcol=3,
#                      )
# campaign.store_data_to_hdf5(data, file_path='gpr_al_6_val.hdf5')

# 7) Case from 8 flux tube GEM UQ campaign (4 parameters, tensor product of grid with 2 points per DoF)
#                       and 4 outputs -> in total, output vector of dimensionality 32

# # Saving both te_transp_flux and ti_transp_flux 

# datafile = 'resuq_main_te_transp_flux_all_csldvnei_37.csv' # this file with TE has both ti and te data
# #datafile = 'resuq_main_te_transp_flux_new_abcd1234_1.csv'

# data = load_csv_file(input_file=datafile, # this file has both ti and te data
#                      n_runs=648,
#                      output_dim=4,
#                      std=True,
#                      startcol=3,
#                      )

# data_ft = split_flux_tubes(data, ft_len=81)
# #print(data_ft)

# campaign.store_data_to_hdf5(data, file_path="gem_uq_648_transp_std_tot.hdf5")
# for i in range(len(data_ft)):
#     campaign.store_data_to_hdf5(data_ft[i], file_path=f"gem_uq_648_transp_std_{i}.hdf5")

"""
# Saving ti_transp_flux 
data = load_csv_file(input_file='resuq_main_ti_transp_flux_all_csldvnei_23.csv',
                     n_runs=648,
                     output_dim=2,
                     std=True,
                     startcol=3,
                     )

data_ft = split_flux_tubes(data, ft_len=81)
#print(data_ft)

campaign.store_data_to_hdf5(data, file_path="gem_uq_648_ti_transp_std_tot.hdf5")
for i in range(len(data_ft)):
    campaign.store_data_to_hdf5(data_ft[i], file_path=f"gem_uq_648_ti_transp_std_{i}.hdf5")
"""

# 8) Case from 8 flux tube GEM0 run, having same number of points for every input dimension

#datafile = "gem0_new_data_20231101.csv"
#datafile = "gem0_new_data_20231208.csv"
datafile = 'gem0_new_data_20231215.csv'

runs_per_ft = 5**4

data = load_csv_to_dict(input_file=datafile,)

data_ft = split_flux_tubes(data, ft_len=runs_per_ft)

campaign.store_data_to_hdf5(data, file_path="gem0_5000_transp_tot_20231216.hdf5")
for i in range(len(data_ft)):
    campaign.store_data_to_hdf5(data_ft[i], file_path=f"gem0_5000_transp_{i}_20231216.hdf5")

