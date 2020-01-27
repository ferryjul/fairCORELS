import csv
import numpy as np
import pandas as pd 
from six.moves import xrange
import os

import argparse
import csv

from ndf import is_pareto_efficient


# parser initialization
parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--dataset', type=int, default=1, help='1: adult_no_relationship_without, 2: compas_without, 3: adult_no_relationship_with, 4:compas_with')
parser.add_argument('--metric', type=int, default=1, help='Fairness metric. 1: SP, 2:  PP, 3: PE, 4: EOpp, 5: EOdds, 6: CUAE')
parser.add_argument('--strat', type=int, default=1, help='Search strategy. 1: bfs, 2:curious, 3: lower_bound, 4: bfs_objective_aware')



args = parser.parse_args()


dataset = {
    1 : 'adult_without_ulb',
    2 : 'compas_without_ulb'
}

metric = {
    1 : 'statistical_parity',
    2 : 'predictive_parity',
    3 : 'predictive_equality',
    4 : 'equal_opportunity',
    5 : 'equalized_odds',
    6 : 'conditional_use_accuracy_equality'
}

strategy = {
    1 : 'bfs',
    2 : 'curious',
    3 : 'lower_bound',
    4 : 'bfs_objective_aware'
}

#save direcory
save_dir = "./pareto/{}/{}".format(dataset[args.dataset], strategy[args.strat])
os.makedirs(save_dir, exist_ok=True)



def compute_front(input_file, output_file):

    errors=[]
    unfairness=[]

    pareto_input = []

    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            error = 1.0 - float(row['accuracy_test'])
            unf = float(row['unfairness_test'])
            errors.append(error)
            unfairness.append(unf)
            pareto_input.append([error, unf])

    pareto_input = np.array(pareto_input)

    msk = is_pareto_efficient(pareto_input)

    

    df = pd.DataFrame()
                
    df['error']      =  [errors[i] for i in xrange(len(errors)) if msk[i]]
    df['unfairness'] =  [unfairness[i] for i in xrange(len(errors)) if msk[i]]

    df.to_csv(output_file, encoding='utf-8', index=False)


input_file = "./results/{}/{}/{}.csv".format(dataset[args.dataset], strategy[args.strat], metric[args.metric])


output_file='{}/{}.csv'.format(save_dir, metric[args.metric])

compute_front(input_file, output_file)