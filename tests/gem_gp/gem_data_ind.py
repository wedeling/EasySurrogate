import numpy as np
import csv
import pandas as pd
import sys
import datetime

import easysurrogate as es

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['te_transp_flux', 'ti_transp_flux', 'te_transp_flux_std', 'ti_transp_flux_std']

def load_csv_to_dict(input_file='gem_data_625.txt', features_names=features_names, target_names=target_names, n_runs=625, input_dim=4, output_dim=2, std=False):
    """
    Loads a CSV file with GEM/GEM0 data contatining columns for input and ouputs into a dictionary
    """

    data = pd.read_csv(input_file, sep=',')
    data = data[[*features_names, *target_names[:output_dim], 'ft']]
    data = data.to_dict(orient='list')
    data = {k:np.array(v).reshape(-1,1) for k,v in data.items()}

    data['ft'] = data['ft'].reshape(-1)

    return data

def split_flux_tubes(data_dict, ft_len=625, n_ft=8, option='column'):
    """
    Split resulting dictionary into multiple dictionaries, one for each flux tube.
    Data separation between different flux tubes can be defined by:
        - fixed number of runs per flux tube
        - array of first runs for each flux tube
    """

    n_tot = data['ti_value'].size # unhardcode key
    if not n_ft:
        n_ft = n_tot // ft_len

    #print(data_dict)###DEBUG

    # Option 1: make a dictionary with keys being differeny flux tube strings and values being dictionaries 
    #    - current storing function does not support tree-like dictionaries
    # data_dict_ft = {}    
    # for i in range(n_ft):

    #     # Option a: readings are ordered by flux tube number, and there is the same number of rows per flux tube
    #     #data_dict_ft['ft'+str(i+1)] = {k:np.array(v[i*ft_len:(i+1)*ft_len]) for (k,v) in data_dict.items()}
        
    #     # Option b: there is an 'ft' column in the data
    #     mask = [data_dict['ft']==i][0][:][:]
    #     print(mask) ###DEBUG
    #     data_dict_ft['ft'+str(i+1)] = {k:v[mask, :] for (k,v) in data_dict.items() if k!='ft'}

    # Option 2: 
    #   a. multiple files for flux tubes
    #   b. multiple quantity names
    #   c. each field with an array for different location, pad with default values
    #   d. each field with an array for different location, pad with None
    data_dict_list = []
    for i in range(n_ft):

        # Option a: readings are ordered by flux tube number, and there is the same number of rows per flux tube
        #data_dict_list.append({k:np.array(v[i*ft_len:(i+1)*ft_len]) for (k,v) in data_dict.items()}) #assume that entries are ordered by flux tube number and there is the same number of rows per flux tube

        # Option b: there is an 'ft' column in the data
        if option == 'column':
            mask = [data_dict['ft']==i][0][:][:]
            data_dict_list.append({k:np.array(v[mask]) for (k,v) in data_dict.items() if k!='ft'})

    #print(f"dimensions of original arrays: {data_dict['ti_transp_flux_std'].shape} ; and new arrays: {data_dict_list[0]['ti_transp_flux_std'].shape}") ###DEBUG

    return data_dict_list


if __name__ == '__main__':

    if len(sys.argv) < 2 :
        gen_id = '20240202'
    else:
        gen_id = sys.argv[1]

    if len(sys.argv) < 3 :
        sav_id = datetime.now().strftime("%Y%m%d")
    else:  
        sav_id = sys.argv[2]

    code = sys.argv[3] if len(sys.argv)>3 else 'gem0py'

    ### Get data to a hfd5

    # Create an ES campaign
    campaign = es.Campaign(load_state=False)

    datafile = f"{code}_new_{gen_id}.csv"

    #print(f"For surrogate training we are reading reading {datafile}") ###DEBUG

    # # I) Case from 8 flux tube GEM0 run, having same number of points for every input dimension

    # # TODO infere from the original data file!
    # n_params = 4
    # n_p_p_p = int(sys.argv[4]) if len(sys.argv)>4 else 5
    # n_ft = 8

    # runs_per_ft = n_p_p_p**n_params
    # n_samples = n_ft*runs_per_ft

    # data = load_csv_to_dict(input_file=datafile)

    # II) Case from 8 flux tube GEM0 run, same number of points for every input, equilibrium (may be) included
    features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho',] # 'profiles_1d_q', 'profiles_1d_gm3']

    data = load_csv_to_dict(input_file=datafile, features_names=features_names)

    data_cols = data.keys()
    col_names_exception = ['tansp', 'ft']

    #features_names = [feat for feat in data_cols if all([x not in feat for x in col_names_exception])]

    n_params = len(features_names)
    n_p_p_p = int(sys.argv[4]) if len(sys.argv)>4 else 5 # could be done via the DataFrame

    n_ft = len(np.unique(data['ft']))
    runs_per_ft = n_p_p_p**n_params
    n_samples = n_ft*runs_per_ft

    ###
    data_ft = split_flux_tubes(data, ft_len=runs_per_ft, n_ft=n_ft)

    print()
    campaign.store_data_to_hdf5(data, file_path=f"{code}_{n_samples}_transp_tot_{sav_id}.hdf5")
    for i in range(len(data_ft)):
        campaign.store_data_to_hdf5(data_ft[i], file_path=f"{code}_{n_samples}_transp_{i}_{sav_id}.hdf5")

    # return number of samples id data set - can break things in multiple ways!
    sys.exit(n_samples)
