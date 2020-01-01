#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --array=1,4,5
#SBATCH --cpus-per-task=48
#SBATCH --mem=503G  
#SBATCH --nodes=1

#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


cd ../experiments

python experiments_fast.py --id=6 --metric=$SLURM_ARRAY_TASK_ID --attr=1

