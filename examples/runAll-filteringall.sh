#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=7000
#SBATCH --time=01:00:00
#SBATCH --job-name=evCPFiltering
#SBATCH -o slurmout_%A.out
#SBATCH -e slurmout_%A.errarray
#SBATCH --array=0-39

python3 newExample-compas.py --epsilon=${SLURM_ARRAY_TASK_ID} --filteringMode=2
python3 newExample-compas.py --epsilon=${SLURM_ARRAY_TASK_ID} --filteringMode=2 --maxTime=30
python3 newExample-compas.py --epsilon=${SLURM_ARRAY_TASK_ID} --filteringMode=2 --maxTime=60
python3 newExample-compas.py --epsilon=${SLURM_ARRAY_TASK_ID} --filteringMode=2 --maxTime=120
