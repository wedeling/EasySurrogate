#!/bin/sh

# # 1) first try data
# datadate=20240110
# modeldate=20240110
# datenow=20240115

# # 2) reverting and trying out full grid data
# datadate=20231208
# modeldate=20240115
# datenow=20240115

# # 3) revert to the full grid data with corrected input locations (2023.12.15/16)
# datadate=20231216
# modeldate=20240117
# datenow=20240117

# # 4) generating new full product pyGEM0 data on transp grid (2024.01.22)
# datadate=20240123
# modeldate=20240123
# datenow=20240123

# # 5) generating new LHC (12k samples) pyGEM0 dataset data on transp grid (2024.01.25)
# datadate=20240125
# modeldate=20240125
# datenow=20240125

# # 6) generating new full tensor product (5k samples) pyGEM0 dataset data, exact on transp grid (2024.01.26)
# datadate=20240126
# modeldate=20240126
# datenow=20240126

# # 7) generating new full tensor product + some inputs (5008 samples) pyGEM0 dataset data (2024.01.24)
# datadate=20240129
# modeldate=20240129
# datenow=20240129

# 8) generating new full tensor product around a new point (500 samples) pyGEM0 dataset data (2024.01.31)

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

#...
nft=8

modeltype='gpr'
codenameshort='gem0'
codename=${codenameshort}'py'

# reinstall the ES package
cd ../..
pip install .
cd tests/gem_gp

# read the CSV files (if needed)
cp ../../../MFW/uq/basicda/${codename}_new_${data_id}.csv ./
python gem_data.py ${data_id} ${data_id}

# train and test the models
for((i=0;i<${nft};i++)); do python train_model.py $i ${data_id} ${model_id} ;done

for((i=0;i<${nft};i++)); do python test_model.py $i ${model_id} ${data_id} ${curr_id} ;done

# save the results
for((i=0;i<${nft};i++)); do cp scan_${i}.csv ../../../MFW/uq/basicda/scan_${codename}${modeltype}_${curr_id}_ft${i}.csv ;done

# save the cut locations
# TODO merge scans!
input_names=('te_value' 'ti_value' 'te_ddrho' 'ti_ddrho')
for name in ${input_names[@]}; do
    for((i=0;i<${nft};i++)); do
        #cp scan_${codename}${modeltype}_remainder_*_ft[0-9].csv ../../../MFW/uq/basicda/
        cp scan_${codenameshort}${modeltype}_remainder_${name}_ft${i}.csv ../../../MFW/uq/basicda/scan_${codename}${modeltype}_remainder_${name}_${curr_id}_ft${i}.csv ;
    done ;
done

#for name in ${input_names[@]}; do for((i=0;i<8;i++)); do cp scan_gem0gpr_remainder_${name}_20230122_ft${i}.csv gpr_scan_20240122_2/scan_gem0gpr_remainder_${name}_20240122_ft${i}.csv ; done ; done

# save the results of test script
savediranme=${modeltype}_scan_${curr_id}_0
mkdir ${savediranme}/ 

cp scan_[0-9].csv ${savediranme}/ 
cp scan_${codenameshort}${modeltype}_remainder_*_ft[0-9].csv ${savediranme}/ 
cp res_[0-9]_o[0-9].csv ${savediranme}/ 
mv pred_vs_orig_[0-9]_o[0-9].pdf ${savediranme}/ 
mv GP_prediction_[0-9]_rand_*_[0-9]_o[0-9].pdf ${savediranme}/ 
mv scan_i[0-9]o[0-9]f[0-9].pdf ${savediranme}/ 
mv gp_abs_err_.pdf ${savediranme}/ 

tar -czvf ${savediranme}.tar.gz ${savediranme}/ 
mv ${savediranme}.tar.gz ../../..

# save the surrogate for the workflow
cd ../../../MFW/workflows/
mkdir surr_model_bckp_${curr_id}
mv surrogate_for_workflow/gem*es*model*pickle surr_model_bckp_${curr_id}
for((i=0;i<${nft};i++)); do
    cp ../../EasySurrogate/tests/gem_gp/model_${codename}_val_scikit-learngaussianRBF_transp_${i}_${curr_id}.pickle surrogate_for_workflow/${codenameshort}_es_model_${i}.pickle ;
done
