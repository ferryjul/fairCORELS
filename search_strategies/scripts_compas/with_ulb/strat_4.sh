#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --array=1
#SBATCH --ntasks=60
#SBATCH --mem-per-cpu=6G

#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL

cd ../..
srun python bench_ulb.py --dataset=8 --metric=$SLURM_ARRAY_TASK_ID --ulb=1 --strat=4

