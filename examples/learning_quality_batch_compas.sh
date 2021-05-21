#!/bin/bash

#SBATCH -n 5
#SBATCH --mem-per-cpu=4500
#SBATCH --time=00:30:00
#SBATCH --job-name=learning
#SBATCH -o slurmout_%A.out
#SBATCH -e slurmout_%A.errarray
#SBATCH --constraint=broadwell
#SBATCH --array=0-383

srun -W 1200 -n 5 python3 learning_quality_compas.py --expe=${SLURM_ARRAY_TASK_ID}
                                                                                                                 
