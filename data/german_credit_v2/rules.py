import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import fpgrowth


from collections import Counter
from imblearn.under_sampling import NearMiss


def rules():
    # sensitive attribute
    

    age = ["age:<25", "age:>=25"]

    marital = ["marital_status:female:divorced/married", "marital_status:male:divorced", "marital_status:male:married", "marital_status:male:single"]

    
    dataset= pd.read_csv("./german_credit_discretized.csv")
    
    y = dataset.credit_rating.values

    df_age = dataset[age]
    df_marital = dataset[marital]
    
    #dropList = ["credit_rating"] + age + marital 
    dropList = ["credit_rating"] + age 
    dataset.drop(labels=dropList, axis=1, inplace=True)

    print('ones rules -->>>>>>>>', len(list(dataset)))

    """
    ll = fpgrowth(dataset, min_support=0.35, max_len=2, use_colnames=True)


    rules = [list(x) for x in ll['itemsets']]

    df_rules = pd.DataFrame()

    print('mined rules -->>>>>>>>', len(rules))

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
        
    """
    #df_all = pd.concat([df_age, df_marital, dataset, df_rules], axis=1)
    #df_all = pd.concat([df_age, dataset, df_rules], axis=1)
    df_all = pd.concat([df_age, dataset], axis=1)
    columns = list(df_all)

    #all data
    df_all['credit_rating'] = y

    print('all rules -->>>>>>>>', len(list(df_all)))

    #saving
    df_all.to_csv("./german_credit_v2_rules_full.csv", encoding='utf-8', index=False)

rules()
