#!/bin/bash
#SBATCH --time=03:00:00

#SBATCH --array=1-6

#SBATCH --cpus-per-task=48

#SBATCH --mem=100G  

#SBATCH --nodes=1 



#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


cd ../experiments

python experiments_fast.py --id=1 --metric=$SLURM_ARRAY_TASK_ID --attr=2

