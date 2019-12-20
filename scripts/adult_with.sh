#!/bin/bash
#SBATCH --time=02:00:00

#SBATCH --array=1-6

#SBATCH --cpus-per-task=20

#SBATCH --mem=25G  

#SBATCH --nodes=5



#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


cd ../experiments

python experiments_fast.py --id=1 --metric=$SLURM_ARRAY_TASK_ID --attr=1  

