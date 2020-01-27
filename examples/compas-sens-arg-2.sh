#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --array=1-6
#SBATCH --mem-per-cpu=6000
#SBATCH --cpus-per-task=15
#SBATCH --nodes=1


python evalStrats.py --strat=2 --forbidSensAttr=0 --metric=$SLURM_ARRAY_TASK_ID

