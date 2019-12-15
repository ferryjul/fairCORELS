#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=def-gambsseb
echo 'Hello, world!' > demo.txt
sleep 30


#SBATCH --mail-user=a.u.matchi@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
