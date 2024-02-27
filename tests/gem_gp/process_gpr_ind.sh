#!/bin/sh

# -) Generating new full tensor product around a new point (500 samples) pyGEM0 dataset data 

if [[ $# -eq 0 ]] ; then
    data_id=20240206
    model_id=20240206
    curr_id=20240206
else
    data_id=$1
    model_id=$1
    curr_id=$1
fi

itnum=${2:-1}

data_id=${data_id}_${itnum}
model_id=${model_id}_${itnum}
curr_id=${curr_id}_${itnum}

locdir=${3:-$(pwd)}

#...
nft=8

# TODO introduce useequil boolean?
num_p_per_param=3

modeltype='gpr'
codenameshort='gem0'
codename=${codenameshort}'py'

# reinstall the ES package
RESINSTALLES=0
#easysurrogatedir='~/code/EasySurrogate/'
easysurrogatedir='/u/yyudin/code/EasySurrogate/'
if [ "$RESINSTALLES" -ne 0 ] ; then
    cd ${easysurrogatedir}
    pip install -e .
fi
cd ${locdir}/

input_names=('te_value' 'ti_value' 'te_ddrho' 'ti_ddrho') # 'profiles_1d_q' 'profiles_1d_gm3')
nparams=${#input_names[@]}
nsamples=$(( ${nft}*${num_p_per_param}**${nparams} ))

# read the CSV files (if needed)
traindatadir='~/code/MFW/uq/basicda'
#cp ${traindatadir}/${codename}_new_${data_name_prefix}.csv ./ # NOT NEEDED, SHOULD BE DONE IN PARENT SCRIPT
python gem_data_ind.py ${data_id} ${data_id} ${codename} ${num_p_per_param}

# train and test the models
for((i=0;i<${nft};i++)); do python train_model_ind.py ${i} ${data_id} ${model_id} ${codename} ${nsamples} ${nparams} ; done

# Next is NOT NEEDED here, but ideally should also return some quality quantification for a surrogate
#for((i=0;i<${nft};i++)); do python test_model_ind.py ${i} ${model_id} ${data_id} ${curr_id} ; done

# save the results (of the scan) and the cut locations - not done here, look up the older script!

# save the results of test script
savediranme=${modeltype}_scan_${curr_id}_0
mkdir ${savediranme}/ 

mv scan_[0-9].csv ${savediranme}/ 
mv scan_${codenameshort}${modeltype}_remainder_*_ft[0-9].csv ${savediranme}/ 
mv res_[0-9]_o[0-9].csv ${savediranme}/ 
mv pred_vs_orig_[0-9]_o[0-9].pdf ${savediranme}/ 
mv GP_prediction_[0-9]_rand_*_[0-9]_o[0-9].pdf ${savediranme}/ 
mv scan_i[0-9]o[0-9]f[0-9].pdf ${savediranme}/ 
mv gp_abs_err_.pdf ${savediranme}/ 

# tar -czvf ${savediranme}.tar.gz ${savediranme}/ 
# mv ${savediranme}.tar.gz ../../..

# save the surrogate for the workflow - prepare for M3-WF run (/muscle3/ dir does not exist yet)
simdirloc=${4:-${locdir}'/../'}
simdirloc=$( realpath ${simdirloc} )
cd ${simdirloc}
mkdir forworkflow

#mkdir surr_model_bckp_${curr_id}
#mv surrogate_for_workflow/gem*es*model*pickle surr_model_bckp_${curr_id}

for((i=0;i<${nft};i++)); do
    cp ${locdir}/model_${codename}_val_scikit-learngaussianRBF_transp_${i}_${curr_id}.pickle ${simdirloc}/forworkflow/${codenameshort}_es_model_${i}.pickle ;
done
