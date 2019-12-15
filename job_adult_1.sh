#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=3
#SBATCH --mincpus=72
#SBATCH --mem-per-cpu=3072M    
#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL

cd experiments

python experiments_parallel.py --id=1 --metric=1
    
