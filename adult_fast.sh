#!/bin/bash
#SBATCH --time=03:00:00

#SBATCH --cpus-per-task=48

#SBATCH --mem=100G  

#SBATCH --nodes=1 



#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


cd experiments

python experiments_fast.py --id=2 --metric=1 --attr=1   

