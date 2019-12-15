#! /bin/bash

datasets=( 1 2 3 4 )
metrics=( 1 2 3 4 5 6 )

for dataset in ${datasets[@]}; do 
    for fairness in ${metrics[@]}; do 
            python3 experiments.py --id=$dataset --metric=$fairness &
    done
done
