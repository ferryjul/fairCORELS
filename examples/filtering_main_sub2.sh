#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4500
#SBATCH --time=02:00:00
#SBATCH --job-name=filtering
#SBATCH -o slurmout_%A.out
#SBATCH -e slurmout_%A.errarray
#SBATCH --constraint=broadwell
#SBATCH --array=0-959

python3 filtering_eval_auto_sub2.py --expe=${SLURM_ARRAY_TASK_ID} --maxTime=120
python3 filtering_eval_auto_sub2.py --expe=${SLURM_ARRAY_TASK_ID} --maxTime=300
python3 filtering_eval_auto_sub2.py --expe=${SLURM_ARRAY_TASK_ID} --maxTime=400
python3 filtering_eval_auto_sub2.py --expe=${SLURM_ARRAY_TASK_ID} --maxTime=500
python3 filtering_eval_auto_sub2.py --expe=${SLURM_ARRAY_TASK_ID} --maxTime=600
python3 filtering_eval_auto_sub2.py --expe=${SLURM_ARRAY_TASK_ID} --maxTime=900
python3 filtering_eval_auto_sub2.py --expe=${SLURM_ARRAY_TASK_ID} --maxTime=1200






