#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mincpus=5
#SBATCH --mem-per-cpu=10G   
#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL

cd experiments

python experiments.py --id=1 --metric=4
    
