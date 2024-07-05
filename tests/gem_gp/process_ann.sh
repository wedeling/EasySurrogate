#!/bin/sh

datadate=20240110
datadate=20231208  #checking old pyGEM0-5k data
datadate=20240321

modeldate=20240110
modeldate=20240116 #latest model
modeldate=20240402

date=20240116
date=20240402

nfts=8

# update ES package
cd ../..
pip install .
cd tests/gem_gp

# train and test the models

for((i=0;i<${nfts};i++)); do python train_model_ann.py ${i} ${date} ${datadate} ;done

for((i=0;i<${nfts};i++)); do python test_model_ann.py ${i} ${modeldate} ${datadate} ;done

# save the results
for((i=0;i<${nfts};i++)); do cp scan_${i}.csv ../../../MFW/uq/basicda/scan_gem0ann_${date}_ft${i}.csv ;done

cp scan_gem0ann_remainder_*_${date}_ft[0-9].csv ../../../MFW/uq/basicda/

postfix="_0"
mkdir annscan_${date}${postfix}

cp scan*csv annscan_${date}${postfix}
mv scan*pdf annscan_${date}${postfix}
mv loss* annscan_${date}${postfix}
tar -czvf annscan_${date}${postfix}.tar.gz annscan_${date}${postfix}
mv annscan_${date}${postfix}.tar.gz ../../..
