#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=def-gambsseb
echo 'Hello, world!'
sleep 30


#SBATCH --mail-user=<aivodji.ulrich@courrier.uqam.ca>
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
