#!/bin/bash
#SBATCH --time=00:2:00
#SBATCH --array=1
#SBATCH --ntasks=60
#SBATCH --mem-per-cpu=2G

#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


srun python bench.py --id=6 --metric=$SLURM_ARRAY_TASK_ID --attr=1 --strat=1

