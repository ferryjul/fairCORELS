#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --array=1-6
#SBATCH --cpus-per-task=32
#SBATCH --mem=70G  
#SBATCH --nodes=1


python evalStrats-adult.py --strat=3 --forbidSensAttr=1 --metric=$SLURM_ARRAY_TASK_ID

