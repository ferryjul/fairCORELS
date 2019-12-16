#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --mincpus=3
#SBATCH --mem-per-cpu=10G   
#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL

cd experiments

python bench.py --id=1 --metric=4
    
