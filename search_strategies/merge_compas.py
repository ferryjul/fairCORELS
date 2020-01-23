import pandas as pd



datasets = ['compas_neg_with_ulb', 'compas_neg_without_ulb']
strategies = ['bfs', 'curious', 'lower_bound', 'bfs_objective_aware']



metrics = ['statistical_parity', 'predictive_parity', 'predictive_equality', 'equal_opportunity', 'equalized_odds', 'conditional_use_accuracy_equality']

datasets_map = {
    'compas_neg_with_ulb': 'COMPAS with ULB',
    'compas_neg_without_ulb': 'COMPAS without ULB'
}

strategies_map = {
    'bfs' : 'BFS original',
    'curious' : 'Curious',
    'lower_bound' : 'Lower bound',
    'bfs_objective_aware' : 'BFS objective-aware'
}

metrics_map = {
    'statistical_parity'                : 'SP',
    'predictive_parity'                 : 'PP',
    'predictive_equality'               : 'PE',
    'equal_opportunity'                 : 'EOpp',
    'equalized_odds'                    : 'EOdds',
    'conditional_use_accuracy_equality' : 'CUAE'
}

df = pd.DataFrame()
filename = './pareto_merged/compas.csv'
for metric in metrics:
    for dataset in datasets:
        for strategy in strategies:
            filename_current = './pareto/{}/{}/{}.csv'.format(dataset, strategy, metric)
            df_current = pd.read_csv(filename_current)
            df_current['dataset'] = [ datasets_map[dataset] for x in df_current['error']]
            df_current['strategy'] = [ strategies_map[strategy] for x in df_current['error']]
            df_current['metric'] = [ metrics_map[metric] for x in df_current['error']]
            df = pd.concat([df, df_current], axis=0)

df.to_csv(filename, encoding='utf-8', index=False)
    