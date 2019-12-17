#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --array=1-6%1
#SBATCH --cpus-per-task=16

#SBATCH --mem=6G  
#SBATCH --nodes=11

#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL

cd ../experiments

python experiments_parallel.py --id=1 --metric=$SLURM_ARRAY_TASK_ID --attr=2
    
