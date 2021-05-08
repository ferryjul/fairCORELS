#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5000
#SBATCH --time=00:15:00
#SBATCH --job-name=filtering
#SBATCH -o slurmout_%A.out
#SBATCH -e slurmout_%A.errarray
#SBATCH --array=0-6719

python3 filtering_eval_auto_sub1.py --expe=${SLURM_ARRAY_TASK_ID}






