#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --array=1-2%1
#SBATCH --cpus-per-task=3

#SBATCH --mem=6G  
#SBATCH --nodes=5

#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL

cd ../experiments

python experiments_parallel.py --id=1 --metric=$SLURM_ARRAY_TASK_ID --attr=2
    
