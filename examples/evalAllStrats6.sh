#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --array=6-6
#SBATCH --mem-per-cpu=10000
#SBATCH --cpus-per-task=15
#SBATCH --nodes=1


python evalStrats.py --strat=$SLURM_ARRAY_TASK_ID

