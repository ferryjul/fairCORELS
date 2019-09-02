import csv
import numpy as np
import pandas as pd 
from six.moves import xrange


def is_pareto_efficient(costs, return_mask = True):
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



def compute_front(df, metric, output_file, test_set=True):

    pareto_input = []
    df=df[df['metric']==metric]

    for _, row in df.iterrows():
        unfairness, error = None, None
        if (test_set):
            unfairness, error = float(row['unfairness_test']), 1.0 - float(row['acc_test'])
        else:
            unfairness, error = float(row['unfairness_train']), 1.0 - float(row['acc_train'])
        
        pareto_input.append([error, unfairness])

    pareto_input = np.array(pareto_input)

    msk = is_pareto_efficient(pareto_input)

    output_file = output_file + 'fairness_' + str(metric) + '.csv'

    

    df[msk].to_csv(output_file, encoding='utf-8', index=False)


#eps 
df_eps = pd.read_csv('./output/df_eps.csv')
df_eps_train_file='./graphs/data/ndf_eps_train_'
df_eps_test_file='./graphs/data/ndf_eps_test_'

compute_front(df_eps, 1, df_eps_train_file, test_set=False)
compute_front(df_eps, 1, df_eps_test_file, test_set=True)

compute_front(df_eps, 2, df_eps_train_file, test_set=False)
compute_front(df_eps, 2, df_eps_test_file, test_set=True)

compute_front(df_eps, 3, df_eps_train_file, test_set=False)
compute_front(df_eps, 3, df_eps_test_file, test_set=True)

compute_front(df_eps, 4, df_eps_train_file, test_set=False)
compute_front(df_eps, 4, df_eps_test_file, test_set=True)


#betas
df_betas = pd.read_csv('./output/df_betas.csv')
df_betas_train_file='./graphs/data/ndf_betas_train_'
df_betas_test_file='./graphs/data/ndf_betas_test_'

compute_front(df_betas, 1, df_betas_train_file, test_set=False)
compute_front(df_betas, 1, df_betas_test_file, test_set=True)

compute_front(df_betas, 2, df_betas_train_file, test_set=False)
compute_front(df_betas, 2, df_betas_test_file, test_set=True)

compute_front(df_betas, 3, df_betas_train_file, test_set=False)
compute_front(df_betas, 3, df_betas_test_file, test_set=True)

compute_front(df_betas, 4, df_betas_train_file, test_set=False)
compute_front(df_betas, 4, df_betas_test_file, test_set=True)