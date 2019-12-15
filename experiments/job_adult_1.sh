#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mincpus=10
#SBATCH --mem-per-cpu=3072M    
#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


python experiments.py --id=1 --metric=1
    
