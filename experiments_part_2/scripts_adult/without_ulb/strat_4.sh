#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --array=1-6
#SBATCH --ntasks=50
#SBATCH --mem-per-cpu=4G

#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL

cd ../..
srun python bench_ulb.py --dataset=1 --metric=$SLURM_ARRAY_TASK_ID --ulb=0 --strat=4

