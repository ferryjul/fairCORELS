#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=9000
#SBATCH --time=04:00:00
#SBATCH --job-name=evCPFiltering
#SBATCH -o slurmout_%A.out
#SBATCH -e slurmout_%A.errarray
#SBATCH --array=0-71

python newExample-compas.py --epsilon=${SLURM_ARRAY_TASK_ID} --uselb=1 --metric=1


