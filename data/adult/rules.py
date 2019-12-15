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
    

    

    dropList = ["income"] + race + gender + country + relationship + marital_status 
    dataset.drop(labels=dropList, axis=1, inplace=True)

    ll = fpgrowth(dataset, min_support=0.06, max_len=2, use_colnames=True)

    print(len(ll))

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
        

    df_all = pd.concat([df_gender, df_marital, dataset, df_rules], axis=1)
    columns = list(df_all)

    """
    #undersampling
    nm3 = NearMiss(version=2, random_state=42, n_jobs=-1)
    X_resampled, y_resampled = nm3.fit_resample(df_all, y)
    print('NearMiss ------', sorted(Counter(y_resampled).items()))
    df_all_undersampled = pd.DataFrame(X_resampled, columns=columns)
    df_all_undersampled['income'] = y_resampled
    """

    #all data
    df_all['income'] = y

    print('-->>>>>>>>', len(list(df_all)))

    #saving
    df_all.to_csv("./adult_rules_full.csv", encoding='utf-8', index=False)
    #df_all_undersampled.to_csv("./data/adult_rules_undersampled.csv", encoding='utf-8', index=False)

rules()