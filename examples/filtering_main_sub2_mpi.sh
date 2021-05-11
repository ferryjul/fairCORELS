#!/bin/bash

#SBATCH --n 20
#SBATCH --mem-per-cpu=4500
#SBATCH --time=02:00:00
#SBATCH --job-name=filtering
#SBATCH -o slurmout_%A.out
#SBATCH -e slurmout_%A.errarray
#SBATCH --constraint=broadwell
#SBATCH --array=0-1

srun python3 filtering_eval_auto_sub2_tentativeMPI.py --expe=${SLURM_ARRAY_TASK_ID} --maxTime=120
srun python3 filtering_eval_auto_sub2_tentativeMPI.py --expe=${SLURM_ARRAY_TASK_ID} --maxTime=300
srun python3 filtering_eval_auto_sub2_tentativeMPI.py --expe=${SLURM_ARRAY_TASK_ID} --maxTime=400
srun python3 filtering_eval_auto_sub2_tentativeMPI.py --expe=${SLURM_ARRAY_TASK_ID} --maxTime=500
srun python3 filtering_eval_auto_sub2_tentativeMPI.py --expe=${SLURM_ARRAY_TASK_ID} --maxTime=600
srun python3 filtering_eval_auto_sub2_tentativeMPI.py --expe=${SLURM_ARRAY_TASK_ID} --maxTime=900
srun python3 filtering_eval_auto_sub2_tentativeMPI.py --expe=${SLURM_ARRAY_TASK_ID} --maxTime=1200






