#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4500
#SBATCH --time=00:35:00
#SBATCH --job-name=filtering
#SBATCH -o slurmout_%A.out
#SBATCH -e slurmout_%A.errarray
#SBATCH --array=0-1680

python3 filtering_eval_auto_sub2.py --expe=${SLURM_ARRAY_TASK_ID}






