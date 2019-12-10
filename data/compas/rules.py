import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import fpgrowth


from collections import Counter
from imblearn.under_sampling import NearMiss


def rules():
    # sensitive attribute
    
    gender = ["gender_Female", "gender_Male"]

    race = ["race_African-American", "race_Caucasian", "race_Asian", "race_Hispanic", "race_Native-American", "race_Other"]

    
    dataset= pd.read_csv("./compas_discretized.csv")
    
    y = dataset.two_year_recid.values

    df_gender = dataset[gender]
    df_race = dataset[race]
    
    dropList = ["two_year_recid"] + race 
    dataset.drop(labels=dropList, axis=1, inplace=True)

    ll = fpgrowth(dataset, min_support=0.05, max_len=2, use_colnames=True)


    rules = [list(x) for x in ll['itemsets']]

    df_rules = pd.DataFrame()

    print(len(rules))

    for rule in rules:
        if (len(rule)==1):
            #key = rule[0]
            #df_rules[key] = dataset[key]
            pass

        else:
            key1 = rule[0]
            key2 = rule[1]

            key = key1 + '__AND__' + key2
            df_rules[key] = np.logical_and(dataset[key1], dataset[key2]).astype(int)
        

    df_all = pd.concat([df_race, dataset, df_rules], axis=1)
    columns = list(df_all)

    #all data
    df_all['two_year_recid'] = y

    print('-->>>>>>>>', len(list(df_all)))

    #saving
    df_all.to_csv("./compas_rules_full.csv", encoding='utf-8', index=False)

rules()

