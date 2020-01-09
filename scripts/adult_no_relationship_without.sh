#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --array=1-6
#SBATCH --cpus-per-task=32
#SBATCH --mem=251G  
#SBATCH --nodes=1



cd ../experiments

python experiments_fast.py --id=6 --metric=$SLURM_ARRAY_TASK_ID --attr=1

