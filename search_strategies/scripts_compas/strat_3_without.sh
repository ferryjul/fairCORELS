#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --array=1-6
#SBATCH --ntasks=60
#SBATCH --mem-per-cpu=5G

#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL

cd ..
srun python bench.py --dataset=2 --metric=$SLURM_ARRAY_TASK_ID --attr=1 --strat=3

