#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --array=1-6
#SBATCH --ntasks=60
#SBATCH --mem-per-cpu=5G

#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


srun python bench.py --dataset=6 --metric=$SLURM_ARRAY_TASK_ID --attr=2 --strat=3

