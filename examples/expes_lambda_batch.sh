#!/bin/bash

#SBATCH -n 5
#SBATCH --mem-per-cpu=4500
#SBATCH --time=30:00:00
#SBATCH --job-name=lambdaExpes
#SBATCH -o slurmout_%A.out
#SBATCH -e slurmout_%A.errarray
#SBATCH --constraint=broadwell
#SBATCH --array=0-671

srun -W 1500 -n 5 python3 expes_lambda_4_datasets.py --expe=${SLURM_ARRAY_TASK_ID}
                                                                                                                 
