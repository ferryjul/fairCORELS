#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --array=4-5
#SBATCH --mem-per-cpu=25000
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1


python evalStrats-adult.py --strat=$SLURM_ARRAY_TASK_ID

