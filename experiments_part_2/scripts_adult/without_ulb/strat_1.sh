#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --array=1
#SBATCH --ntasks=50
#SBATCH --mem-per-cpu=10G

#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL

cd ../..
srun python bench_ulb.py --dataset=1 --metric=$SLURM_ARRAY_TASK_ID --ulb=0 --strat=1
