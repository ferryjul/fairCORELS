#!/bin/bash
#SBATCH --time=00:10:00

#SBATCH --cpus-per-task=12

#SBATCH --mem=6G  



#SBATCH --account=def-gambsseb
#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=ALL


python demo.py
    
