#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --array=1-6
#SBATCH --cpus-per-task=11
#SBATCH --mem=20G  


#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL

cd ../experiments

python experiments.py --id=2 --metric=$SLURM_ARRAY_TASK_ID --attr=2
    