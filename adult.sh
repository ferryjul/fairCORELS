#!/bin/bash
#SBATCH --time=05:00:00

#SBATCH --cpus-per-task=48

#SBATCH --mem=80G  

#SBATCH --nodes=1 



#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


cd experiments

python experiments_parallel.py --id=1 --metric=1 --attr=1   
