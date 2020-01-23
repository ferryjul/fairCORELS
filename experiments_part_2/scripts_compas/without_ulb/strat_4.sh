#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --array=1,4,5
#SBATCH --ntasks=32
#SBATCH --mem-per-cpu=6G

#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL

cd ../..
srun python bench_ulb.py --dataset=8 --metric=$SLURM_ARRAY_TASK_ID --ulb=0 --strat=4

