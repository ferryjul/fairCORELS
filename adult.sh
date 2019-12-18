#!/bin/bash
#SBATCH --time=00:20:00

#SBATCH --cpus-per-task=17

#SBATCH --mem=5G  

#SBATCH --nodes=2



#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


cd experiments

python experiments_parallel.py --id=1 --metric=1 --attr=2    
