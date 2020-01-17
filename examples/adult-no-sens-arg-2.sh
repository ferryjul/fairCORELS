#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --array=1-6
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G  
#SBATCH --nodes=1


python evalStrats-adult.py --strat=2 --forbidSensAttr=1 --metric=$SLURM_ARRAY_TASK_ID

