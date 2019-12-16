#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=20G  
#SBATCH --cpus-per-task=10
#SBATCH --nodes=10

#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL

cd experiments

python experiments.py --id=1 --metric=1
    





