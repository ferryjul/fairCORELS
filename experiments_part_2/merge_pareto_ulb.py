import pandas as pd



datasets = ['adult_no_relationship_neg_without', 'compas_neg_without']
strategies = ['withUnfairnessLB', 'withoutUnfairnessLB']


metrics = ['statistical_parity', 'predictive_parity', 'predictive_equality', 'equal_opportunity', 'equalized_odds', 'conditional_use_accuracy_equality']

datasets_map = {
    'adult_no_relationship_neg_without': 'Adult',
    'compas_neg_without': 'COMPAS'
}

strategies_map = {
    'withUnfairnessLB' : 'with ULB',
    'withoutUnfairnessLB' : 'without ULB'
}




for metric in metrics:
    for dataset in datasets:
        df = pd.DataFrame()
        filename = './pareto_merged_ulb/{}_{}.csv'.format(datasets_map[dataset],metric)
        for strategy in strategies:
            filename_current = './pareto_ulb/{}/{}/{}.csv'.format(dataset, strategy, metric)
            df_current = pd.read_csv(filename_current)
            df_current['strategy'] = [strategies_map[strategy] for x in df_current['error']]
            df = pd.concat([df, df_current], axis=0)

        df.to_csv(filename, encoding='utf-8', index=False)
    