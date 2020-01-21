import csv
import numpy as np
import pandas as pd 
from six.moves import xrange
import os

import argparse
import csv

from ndf import is_pareto_efficient


datasets = ['adult_no_relationship_neg_without', 'compas_neg_without']
strategies = ['withUnfairnessLB', 'withoutUnfairnessLB']

metrics = ['statistical_parity', 'predictive_parity', 'predictive_equality', 'equal_opportunity', 'equalized_odds', 'conditional_use_accuracy_equality']





def compute_front(input_file, output_file):

    errors=[]
    unfairness=[]

    pareto_input = []

    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            error = 1.0 - float(row['accuracy'])
            unf = float(row['unfairness'])
            errors.append(error)
            unfairness.append(unf)
            pareto_input.append([error, unf])

    pareto_input = np.array(pareto_input)

    msk = is_pareto_efficient(pareto_input)

    

    df = pd.DataFrame()
                
    df['error']      =  [errors[i] for i in xrange(len(errors)) if msk[i]]
    df['unfairness'] =  [unfairness[i] for i in xrange(len(errors)) if msk[i]]

    df.to_csv(output_file, encoding='utf-8', index=False)


for dataset in datasets:
    for strategy in strategies:
        save_dir = "./pareto_ulb/{}/{}".format(dataset, strategy)
        os.makedirs(save_dir, exist_ok=True)
        for metric in metrics:
            input_file = "./results/results/{}/{}/bfs_objective_aware/{}.csv".format(strategy,dataset, metric)
            output_file='{}/{}.csv'.format(save_dir, metric)
            compute_front(input_file, output_file)