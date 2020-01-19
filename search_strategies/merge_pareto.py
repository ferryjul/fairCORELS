import pandas as pd



datasets = ['adult_no_relationship_without', 'compas_without']
strategies = ['bfs', 'curious', 'lower_bound', 'bfs_objective_aware']



metrics = ['statistical_parity', 'predictive_parity', 'predictive_equality', 'equal_opportunity', 'equalized_odds', 'conditional_use_accuracy_equality']

datasets_map = {
    'adult_no_relationship_without': 'Adult',
    'compas_without': 'COMPAS'
}

strategies_map = {
    'bfs' : 'BFS original',
    'curious' : 'Curious',
    'lower_bound' : 'Lower bound',
    'bfs_objective_aware' : 'BFS objective-aware'
}

for metric in metrics:
    df = pd.DataFrame()
    filename = './pareto_merged/{}.csv'.format(metric)
    for dataset in datasets:
        for strategy in strategies:
            filename_current = './pareto/{}/{}/{}.csv'.format(dataset, strategy, metric)
            df_current = pd.read_csv(filename_current)
            df_current['dataset'] = [ datasets_map[dataset] for x in df_current['error']]
            df_current['strategy'] = [ strategies_map[strategy] for x in df_current['error']]
            df = pd.concat([df, df_current], axis=0)

    df.to_csv(filename, encoding='utf-8', index=False)
    