#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --array=1-6%1
#SBATCH --cpus-per-task=11

#SBATCH --mem=60G  
#SBATCH --nodes=1

#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL

cd ../experiments

python experiments_parallel.py --id=1 --metric=$SLURM_ARRAY_TASK_ID --attr=1
    
