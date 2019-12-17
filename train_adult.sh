#!/bin/bash
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G  


#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL

cd experiments

python experiments.py --id=1 --metric=1
    

