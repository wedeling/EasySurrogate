#!/bin/bash -l

## job name
#SBATCH --job-name=HPO_GPR_

## stdout and stderr files
#SBATCH --output=hpo-gpr-out.%j
#SBATCH --error=hpo-gpr-err.%j

## wall time in format (HOURS):MINUTES:SECONDS
#SBATCH --time=1:00:00

## number of nodes and tasks per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

#SBATCH --partition=medium
###SBATCH --qos=

## grant
#SBATCH --mail-type=ALL
#SBATCH --mail-user=[enter your email]

######################################
# Loading modules
module load anaconda/3/2021.11 

# Python set-up
source activate $HOME/[your environment]

export SYS=COBRA
export SCRATCH=$SCRATCH

# For QCG-PilotJob usage
export ENCODER_MODULES
export EASYPJ_CONFIG=conf.sh

export HPC_EXECUTION=1

echo -e '> In this run: use ExecuteLocal only + QCGPJ pool + '$SLURM_NNODES' nodes /n'

####################################
# Run the training code

python3 train_gp.py > hpo-gpr-log.${SLURM_JOBID}

echo "> Finished a SLURM job for HPO!"