#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-6
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3072M    
#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


python experiments.py --id=1 --metric=$SLURM_ARRAY_TASK_ID 
    
