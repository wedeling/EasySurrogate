#!/bin/sh

date=20231217

# train and test the models
for((i=0;i<8;i++)); do python train_model_ann.py $i ;done
for((i=0;i<8;i++)); do python test_model_ann.py $i ${date} ;done

# save the results
for((i=0;i<8;i++)); do cp scan_${i}.csv ../../../MFW/uq/basicda/scan_gem0surr_${date}_ft${i}.csv ;done
mkdir annscan_${date}
mv scan* annscan_${date}
mv loss* annscan_${date}
tar -czvf annscan_${date}tar.gz annscan_${date}
mv  ${date}annscan_${date}tar.gz ../../..

