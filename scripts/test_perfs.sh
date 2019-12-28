#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=250G  
#SBATCH --nodes=1



#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


cd ../experiments

python test_perfs.py --id=8 --attr=1 --metric=1

