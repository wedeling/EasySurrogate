import numpy as np
import easysurrogate as es

from itertools import product
from datetime import date

np.random.seed(42)

features_names = ['te_value', 'ti_value', 'te_ddrho', 'ti_ddrho']
target_names = ['ti_transp_flux']

qoi_targ=2099023.289881937 #TODO read from production run result csv
print('> Looking for samples to yield: {0}'.format(qoi_targ))

#file_path='model_val_LocGaussianRbf_16012023.pickle'

def find_candidates(features_names, target_names, qoi_targ, model_file, save_file_post):
    """
    Saves a CSV file with a suggested coordinate in input space
    """

    try:
        campaign = es.Campaign(load_state=True, file_path=model_file)
    except OSError:
        print('No such model file!')
        return

    #data_frame = campaign.load_hdf5_data(file_path='gem_uq_81_std.hdf5')

    # # Preparing new features and targets
    #features = [data_frame[k] for k in features_names if k in data_frame]
    #target = np.concatenate([data_frame[k] for k in target_names if k in data_frame], axis=1)

    # Try to retrain on probability of improvement to be closer to a target QoI value using samples of full feature dataset
    X_new = campaign.surrogate.train_sequentially(
                                n_iter=1, # crucial, otherwise will try to start re-training model
                                feats=features_names, 
                                target=qoi_targ, 
                                acquisition_function='poi_sq_dist_to_val',
                                savefile_postfix=save_file_post,
                                                )

    # Test on a single sample
    m_new, v_new = campaign.surrogate.predict(X_new.reshape(-1, 1))

    print('> The new sample in input space is : {0} and the predicted value is : {1}'.format(X_new, m_new[0]))
    
    return m_new[0]

hpo_dict = {
        'implementation': ['Skit', 'Loc'],
        'likelihood': ['Gaussian', 'Student'],
        'kernel' : ['Matern', 'Rbf'],
           }

date_model = '13012023'
date_al = date.today().strftime("%d%m%Y")

for t in product(*[v for _,v in hpo_dict.items()]):
    
    model_file_name = 'model_val_'+ ''.join(list(t)) +'_'+ date_model +'.pickle'
    save_file_post  = ''.join(list(t)) +'_'+ date_al
    
    find_candidates(features_names, target_names, qoi_targ, model_file_name, save_file_post)
