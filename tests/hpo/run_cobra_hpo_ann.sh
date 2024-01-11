#!/bin/bash -l

## job name
#SBATCH --job-name=HPO_ANN_

## stdout and stderr files
#SBATCH --output=hpo-ann-out.%j
#SBATCH --error=hpo-ann-err.%j

## wall time in format (HOURS):MINUTES:SECONDS
#SBATCH --time=4:30:00

## number of nodes and tasks per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

#SBATCH --partition=medium
###SBATCH --qos=

## grant
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yyudin@ipp.mpg.de

######################################
# Loading modules
module load anaconda/3/2021.11 

# Python set-up
source activate $HOME/conda-envs/python3114

export SYS=COBRA
export SCRATCH=$SCRATCH

# For QCG-PilotJob usage
export ENCODER_MODULES
export EASYPJ_CONFIG=conf.sh

export HPC_EXECUTION=1

echo -e '> In this run: use ExecuteLocal only + QCGPJ pool + '$SLURM_NNODES' nodes /n'

####################################
# latest latexplotlib may contain a bug!
pip install latexplotlib

# Update the package
cd ../..
pip install .
cd tests/hpo

# Define the flux tube number
FT_LEN=8


#for((i=0;i<${FT_LEN};i++)); do
for((i=6;i<${FT_LEN};i++)); do

    DATAFILE=gem06_f${i}.hdf5

    # Run the training code
    echo '> Running the training code for flux tube #'${i}' and datafile '${DATAFILE}

    python3 train_ann.py ${i} >> hpo-ann-log.${SLURM_JOBID}

done;

echo '...'
echo "> Finished a SLURM job for HPO!"
