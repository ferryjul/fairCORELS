#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=7000
#SBATCH --time=03:00:00
#SBATCH --job-name=evCPFiltering
#SBATCH -o slurmout_%A.out
#SBATCH -e slurmout_%A.errarray
#SBATCH --array=0-39

python3 newExample-compas-singleOPT.py --epsilon=${SLURM_ARRAY_TASK_ID} --filteringMode=1
python3 newExample-compas-singleOPT.py --epsilon=${SLURM_ARRAY_TASK_ID} --filteringMode=1 --maxTime=60
python3 newExample-compas-singleOPT.py --epsilon=${SLURM_ARRAY_TASK_ID} --filteringMode=1 --maxTime=120
python3 newExample-compas-singleOPT.py --epsilon=${SLURM_ARRAY_TASK_ID} --filteringMode=1 --maxTime=300
python3 newExample-compas-singleOPT.py --epsilon=${SLURM_ARRAY_TASK_ID} --filteringMode=1 --maxTime=120 --policy="objective"
python3 newExample-compas-singleOPT.py --epsilon=${SLURM_ARRAY_TASK_ID} --filteringMode=1 --maxTime=300 --policy="objective"





