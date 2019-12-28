import csv
import numpy as np
import pandas as pd 
from six.moves import xrange


import argparse
import csv

# parser initialization
parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--id', type=int, default=1, help='dataset id: 1 for Adult Income, 2 for Compas, 3 for German Credit and 4 for Default Credit')
parser.add_argument('--m', type=int, default=1, help='fairness metric: 1 statistical_parity, 2 predictive_parity, 3 predictive_equality, 4 equal_opportunity')
parser.add_argument('--attr', type=int, default=1, help='use sensitive attribute: 1 no, 2 yes')
parser.add_argument('--exp', type=str, default='results', help='experiments folder')



args = parser.parse_args()




dataset_dict = {
    1 : 'adult',
    2 : 'compas',
    3 : 'german_credit',
    4 : 'default_credit',
    5 : 'adult_marital',
    6 : 'adult_no_relationship'
}

metric_dict = {
    1 : 'statistical_parity',
    2 : 'predictive_parity',
    3 : 'predictive_equality',
    4 : 'equal_opportunity',
    5 : 'equalized_odds',
    6 : 'conditional_use_accuracy_equality'
}

suffix = {
    1 : 'without_dem',
    2 : 'with_dem',
}



def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

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
                
    df['error'] = [errors[i] for i in xrange(len(errors)) if msk[i]]
    df['unfairness'] = [unfairness[i] for i in xrange(len(unfairness)) if msk[i]]

    df.to_csv(output_file, encoding='utf-8', index=False)


input_file = "./{}/{}_{}/{}.csv".format(args.exp, dataset_dict[args.id], suffix[args.attr], metric_dict[args.m])


output_file='./pareto/{}_{}_{}.csv'.format(dataset_dict[args.id], metric_dict[args.m], suffix[args.attr])

compute_front(input_file, output_file)