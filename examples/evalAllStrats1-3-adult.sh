#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --array=1-3
#SBATCH --mem-per-cpu=11000
#SBATCH --cpus-per-task=15
#SBATCH --nodes=1


python evalStrats-adult.py --strat=$SLURM_ARRAY_TASK_ID

