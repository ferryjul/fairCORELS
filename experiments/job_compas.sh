#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --array=1-6
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3072M    
#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


python experiments.py --id=2 --metric=$SLURM_ARRAY_TASK_ID 
    
