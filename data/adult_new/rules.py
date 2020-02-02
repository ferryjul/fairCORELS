import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import fpgrowth


from collections import Counter


def rules():
    # sensitive attribute
    gender = ["gender_Female", "gender_Male"]

    marital_status = ["maritalStatus_married", "maritalStatus_single"]



    dataset= pd.read_csv("./adult_discretized.csv")
    
    y = dataset.income.values

    df_gender = dataset[gender]
    

    dropList = ["income"] + gender  
    dataset.drop(labels=dropList, axis=1, inplace=True)


    #add neg cols
    cols = list(dataset)
    df_neg = pd.DataFrame()
    

    for col in cols:
        df_neg['not_{}'.format(col)] = 1 - dataset[col]

    print('ones rules -->>>>>>>>', len(list(dataset)) + len(list(df_neg)))


    ll = fpgrowth(dataset, min_support=0.01, max_len=2, use_colnames=True)


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
        

    df_all = pd.concat([df_gender, dataset, df_neg, df_rules], axis=1)
    columns = list(df_all)

    

    #all data
    df_all['income'] = y

    print('all rules -->>>>>>>>', len(list(df_all)))

    #saving
    df_all.to_csv("./adult_new_full.csv", encoding='utf-8', index=False)

rules()
