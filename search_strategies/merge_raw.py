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



args = parser.parse_args()



dataset = {
    1 : 'adult_no_relationship_neg_without_ulb',
    2 : 'compas_neg_without_ulb'
}

metric = {
    1 : 'statistical_parity',
    2 : 'predictive_parity',
    3 : 'predictive_equality',
    4 : 'equal_opportunity',
    5 : 'equalized_odds',
    6 : 'conditional_use_accuracy_equality'
}



strategies = ['bfs', 'curious', 'lower_bound', 'bfs_objective_aware']


strategies_map = {
    'bfs' : 'BFS original',
    'curious' : 'Curious',
    'lower_bound' : 'Lower bound',
    'bfs_objective_aware' : 'BFS objective-aware'
}

#save direcory
save_dir = "./result_merged/{}".format(dataset[args.dataset])
os.makedirs(save_dir, exist_ok=True)



df = pd.DataFrame()
filename = '{}/{}.csv'.format(save_dir, metric[args.metric])

for strategy in strategies:
    filename_current = './results/{}/{}/{}.csv'.format(dataset[args.dataset], strategy, metric[args.metric])
    df_current = pd.read_csv(filename_current)
    df_current['strategy'] = [strategies_map[strategy] for x in df_current['accuracy']]
    df = pd.concat([df, df_current], axis=0)

df.to_csv(filename, encoding='utf-8', index=False)
    