#!/bin/sh

# Copy results of HPO (the best model) into a new subdirectory

MODELDESCRIPTOR=gem0_es_ann_

LASTDATE=20240111

# define destination folder
#NEWFOLDER=20231130_models
#NEWFOLDER=20231215_models
NEWFOLDER=${LASTDATE}_models #ANN run on GEM0 data on same grid

mkdir ${NEWFOLDER}

# define source folders
#FOLDERPREFIX=hpo_easysurrogate_f
FOLDERPREFIX=hpo_easysurrogate_ann_f

# Next could be read from 'ls -d ${FOLDERPREFIX}[0-9] some date...'
#FOLDERSUFFIX='_20231130_14'
#FOLDERSUFFIX='_2023121' #ANN run on GEM0 data on same grid
FOLDERSUFFIX='_'${LASTDATE}'_1' #2024011

#FOLDERS=('2559_wmbdq3l9' '2932__e9q67cj' '3307_dca3cv55' '3636_hmrr9kv5' '4010_7nov_o0r' '4343_3piogul4' '4716_6m7eajmq' '5051__ngeq4gb')
#FOLDERS=('4_115046_faoiidv8' '4_120715__o62xru9' '3_204322_f2mqqw4t' '4_123209_2hlb8yxh' '4_124834_po82zy4m' '4_130458_914zk24y' '4_132121_b1vm8cxy' '4_133747_gdihum57')
#FOLDERS=('8_171046_ooygaf0e' '8_174205_bucn60_s' '8_181329_yjqxti7u' '8_184451_3n2p1x0p' '8_191608_s6v_qq7g' '8_194728_yy8yeex_' '8_201855_0alnuw9m' '9_152226_hksn9jae') #ANN run on GEM0 data on same grid
FOLDERS=('51300_g2lci2kh' '54427_fe0rl_ij' '61548_b67f6lot' '64707_yhl1w5h_' '71832_ib__mazu' 'should be there for f5' 'new' 'new') #20240111

# copy the best model from each folder
# Next could be parsed from a submission log file
#RUNS=('160' '163' '151' '151' '151' '151' '163' '151')
#RUNS=('1794' '1797' '1794' '1797' '1797' '1794' '1797' '1797')
#RUNS=('2792' '2396' '2392' '2796' '2796' '2392' '2392' '2792') #ANN run on GEM0 data on same grid
RUNS=('2730' '2658' '2658' '2738' '2658' '2738' '' '') #20240111

for i in ${!FOLDERS[@]}; do 
  cp ${FOLDERPREFIX}${i}${FOLDERSUFFIX}${FOLDERS[$i]}/runs/run_${RUNS[$i]}/model.pickle ${NEWFOLDER}/${MODELDESCRIPTOR}${i}.pickle
done


# perform new analysis on the best models to generate scans 
TARGETFOLDER=../../../MFW/uq/basicda
cd ../gem_gp

for((i=0;i<8;i++)); do 
  cp ../hpo/${NEWFOLDER}/${MODELDESCRIPTOR}${i}.pickle ./gem0_es_ann_${i}_${LASTDATE}.pickle
  python test_model_ann.py ${i} ${LASTDATE}
  cp scan_${i}.csv ${TARGETFOLDER}/scan_gem0surr_${LASTDATE}_ft${i}.csv
done

