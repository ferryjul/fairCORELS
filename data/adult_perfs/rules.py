import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import fpgrowth


from collections import Counter
from imblearn.under_sampling import NearMiss


def rules():
    # sensitive attribute
    gender = ["gender_Female", "gender_Male"]

    race = ["race_AmerIndianEskimo", "race_AsianPacIslander", "race_Black", "race_Other", "race_White"]

    marital_status = ["maritalStatus_married", "maritalStatus_single"]

    relationship = ["relationship_husband", "relationship_notInFamily", "relationship_otherRelative", 
                        "relationship_ownChild", "relationship_unmarried", "relationship_wife"]

    country = ["nativeCountry_USA", "nativeCountry_notUSA"]



    dataset= pd.read_csv("./adult_discretized.csv")
    
    y = dataset.income.values

    df_gender = dataset[gender]
    df_marital= dataset[marital_status]
    df_race = dataset[race]
    df_country = dataset[country]
    

    

    #dropList = ["income"] + race + gender + country + relationship + marital_status
    #dropList = ["income"] + race + gender + country  + marital_status 
    dropList = ["income"] + race + gender + country  
    dataset.drop(labels=dropList, axis=1, inplace=True)

    print('ones rules -->>>>>>>>', len(list(dataset)))

    ll = fpgrowth(dataset, min_support=0.01, max_len=2, use_colnames=True)

    #print(len(ll))

    rules = [list(x) for x in ll['itemsets']]
    

    df_rules = pd.DataFrame()


    print('mined rules -->>>>>>>>', len(rules))
    

    for rule in rules:
        if (len(rule)==1):
            #key = rule[0]
            #df_rules[key] = dataset[key]
            pass

        elif (len(rule)==2):
            key1 = rule[0]
            key2 = rule[1]

            key = key1 + '__AND__' + key2
            df_rules[key] = np.logical_and(dataset[key1], dataset[key2]).astype(int)

        else :
            key1 = rule[0]
            key2 = rule[1]
            key3 = rule[2]

            key = key1 + '__AND__' + key2 + '__AND__' + key3
            df_rules[key] = np.logical_and(np.logical_and(dataset[key1], dataset[key2]), dataset[key3]).astype(int)
        

    df_all = pd.concat([df_gender, dataset, df_rules], axis=1)
    columns = list(df_all)


    #all data
    df_all['income'] = y

    print('all rules -->>>>>>>>', len(list(df_all)))

    #saving
    df_all.to_csv("./adult_perfs_rules_full.csv", encoding='utf-8', index=False)

rules()