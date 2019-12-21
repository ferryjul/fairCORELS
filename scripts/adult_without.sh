#!/bin/bash
#SBATCH --time=$var_time
#SBATCH --array=$var_array
#SBATCH --cpus-per-task=$var_cpus
#SBATCH --mem=$var_mem 
#SBATCH --nodes=$var_nodes


#SBATCH --account=$var_account
#SBATCH --mail-user=$var_mail
#SBATCH --mail-type=ALL


cd ../experiments

python experiments_fast.py --id=1 --metric=$SLURM_ARRAY_TASK_ID --attr=1

