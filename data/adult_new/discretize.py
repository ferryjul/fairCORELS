from io import StringIO
import numpy as np
import pandas as pd
import math
from collections import Counter
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split


from optbinning import OptimalBinning


raw_features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'maritalStatus', 'occupation', 'relationship', 'race', 'gender', 'capitalGain', 'capitalLoss','hoursPerWeek', 'nativeCountry', 'income']

to_use = ['age', 'workclass',  'education',  'maritalStatus', 'occupation', 'gender', 'capitalGain', 'hoursPerWeek', 'income']
num_cols = ['hoursPerWeek', 'age', 'capitalGain']

dataset = pd.read_csv('./adult_clean.csv')

dataset = dataset[to_use]

df_train, df_disc = train_test_split(dataset, stratify=dataset['income'], test_size=0.75, random_state=42)

def process_bin(x):
    res = ""
    x = str(x).replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
    part_1, part_2 = x.split(',')[0], x.split(',')[1]

    if part_1=='-inf':
        res = '<' + part_2
    elif part_2=='inf':
        res = '>' + part_1
    else:
        res = part_1 + '-' + part_2
    
    return res


for col in num_cols:
    X = df_train[col]
    y = df_train.income    
    optb = OptimalBinning(name=col, dtype="numerical", solver="cp")
    optb.fit(X, y)

    splits = optb.splits
    splits = [-np.inf] + sorted(splits) + [np.inf]

    df_disc[col] = pd.cut(x=df_disc[col], bins=splits)
    df_disc[col] = df_disc[col].apply(process_bin)

y = df_disc['income']
df_disc.drop(labels=['income'], axis = 1, inplace = True) 
df_bin = pd.get_dummies(df_disc)
df_bin = pd.DataFrame(df_bin, columns=list(df_bin))
df_bin['income'] = y

df_bin.to_csv("./adult_discretized.csv", encoding='utf-8', index=False)



