#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=15000
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1


python evalStrats.py --strat=6

