import numpy as np 
import pandas as pd 
import os
import json
import ast



datasets = ["adult_marital_without_dem", "adult_no_relationship_without_dem", "adult_with_dem", "adult_without_dem",
            "compas_with_dem", "compas_without_dem", "default_credit_with_dem", "default_credit_without_dem",
            "german_credit_with_dem", "german_credit_without_dem"]

metrics = {
    1: "statistical_parity",
    2 : "predictive_parity",
    3 : "predictive_equality",
    4 : "equal_opportunity",
    5 : "equalized_odds",
    6 : "conditional_use_accuracy_equality"
}

acc_pos = [0, 5, 10, 15, 20]
unf_pos = [1, 6, 11, 16, 21]


for dataset in datasets:
    os.makedirs('./results_median/{}'.format(dataset), exist_ok=True)
    print(dataset)

    for _, value in metrics.items():

        filename = './results/{}/{}.csv'.format(dataset,value)
        savename = './results_median/{}/{}.csv'.format(dataset,value)

        df = pd.read_csv(filename)

        accuracy = df.accuracy.tolist()
        unfairness = df.unfairness.tolist()
        description = df.models.apply(lambda x: x[1:-1].split(','))

        
        accuracy_median = []
        unfairness_median = []

        for row in description:
            acc = []
            unf = []

            for pos in acc_pos:
                acc.append(float(row[pos].split(':')[1]))

            for pos in unf_pos:
                unf.append(float(row[pos].split(':')[1]))
            
            
                
            accuracy_median.append(np.median(acc))
            unfairness_median.append(np.median(unf))
        
        df['accuracy'] = accuracy_median
        df['unfairness'] = unfairness_median
        df.to_csv(savename, encoding='utf-8', index=False)


    


    
        
        