#!/bin/bash
#SBATCH --time=00:10:00

#SBATCH --cpus-per-task=10

#SBATCH --mem=10G  

#SBATCH --nodes=1  



#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


cd experiments

python experiments_parallel.py --id=1 --metric=1 --attr=2    
