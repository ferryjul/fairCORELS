import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import fpgrowth


from collections import Counter
from imblearn.under_sampling import NearMiss


def rules():
    # sensitive attribute
    #LIMIT_BAL:145000.0_to_365000.0,LIMIT_BAL:365000.0_to_inf,LIMIT_BAL:45000.0_to_145000.0,
    # SEX:female,SEX:male,
    # EDUCATION:grad_school,EDUCATION:high_school,EDUCATION:na,EDUCATION:others,EDUCATION:university,EDUCATION:unknown,
    # MARRIAGE:married,MARRIAGE:na,MARRIAGE:others,MARRIAGE:single,
    # AGE:-inf_to_25.5,AGE:25.5_to_45.5,AGE:45.5_to_inf,
    # PAY_0:delay_1_month,PAY_0:delay_>1_month,PAY_0:no_consumption,PAY_0:pay_duly,PAY_0:revolving_credit,
    # PAY_2:delay_>=1_month,PAY_2:no_consumption,PAY_2:pay_duly,PAY_2:revolving_credit,PAY_3:delay_>=1_month,
    # PAY_3:no_consumption,PAY_3:pay_duly,PAY_3:revolving_credit,PAY_4:delay_>=1_month,PAY_4:no_consumption,
    # PAY_4:pay_duly,PAY_4:revolving_credit,PAY_5:delay_>=1_month,PAY_5:no_consumption,PAY_5:pay_duly,
    # PAY_5:revolving_credit,PAY_6:delay_>=1_month,PAY_6:no_consumption,PAY_6:pay_duly,PAY_6:revolving_credit,
    # BILL_AMT1:-inf_to_54406.0,BILL_AMT1:54406.0_to_inf,BILL_AMT2:21200<_<=64008,BILL_AMT2:2984<_<=21200,
    # BILL_AMT2:<=2984,BILL_AMT2:>64008,BILL_AMT3:20088<_<=60165,BILL_AMT3:2665<_<=20088,BILL_AMT3:<=2665,
    # BILL_AMT3:>60165,BILL_AMT4:19052<_<=54509,BILL_AMT4:2326<_<=19052,BILL_AMT4:<=2326,BILL_AMT4:>54509,
    # BILL_AMT5:1763<_<=18104,BILL_AMT5:18104<_<=50196,BILL_AMT5:<=1763,BILL_AMT5:>50196,BILL_AMT6:1256<_<=17071,
    # BILL_AMT6:17071<_<=49200,BILL_AMT6:<=1256,BILL_AMT6:>49200,PAY_AMT1:-inf_to_21.5,PAY_AMT1:21.5_to_4552.5,
    # PAY_AMT1:23002.0_to_inf,PAY_AMT1:4552.5_to_23002.0,PAY_AMT2:-inf_to_91.0,PAY_AMT2:19635.5_to_inf,
    # PAY_AMT2:4980.5_to_19635.5,PAY_AMT2:91.0_to_4980.5,PAY_AMT3:-inf_to_17.5,PAY_AMT3:17.5_to_4641.5,
    # PAY_AMT3:19981.0_to_inf,PAY_AMT3:4641.5_to_19981.0,PAY_AMT4:-inf_to_0.5,PAY_AMT4:0.5_to_1900.5,
    # PAY_AMT4:1900.5_to_4327.5,PAY_AMT4:4327.5_to_inf,PAY_AMT5:-inf_to_0.5,PAY_AMT5:0.5_to_2000.5,
    # PAY_AMT5:2000.5_to_9986.5,PAY_AMT5:9986.5_to_inf,PAY_AMT6:-inf_to_1.5,PAY_AMT6:1.5_to_2000.5,
    # PAY_AMT6:2000.5_to_9849.5,PAY_AMT6:9849.5_to_inf,
    # default_payment_next_month

    
    gender = ["gender:Female", "gender:Male"]

    marital = ["MARRIAGE:married", "MARRIAGE:na", "MARRIAGE:others", "MARRIAGE:single"]

    
    dataset= pd.read_csv("./default_credit_discretized.csv")
    
    y = dataset.default_payment_next_month.values

    df_gender = dataset[gender]
    df_marital = dataset[marital]
    
    dropList = ["default_payment_next_month"] + gender + marital 

    dataset.drop(labels=dropList, axis=1, inplace=True)

    ll = fpgrowth(dataset, min_support=0.35, max_len=2, use_colnames=True)


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

    #all data
    df_all['default_payment_next_month'] = y

    print('-->>>>>>>>', len(list(df_all)))

    #saving
    df_all.to_csv("./default_credit_rules_full.csv", encoding='utf-8', index=False)

rules()


"""
dataset= pd.read_csv("./default_credit_binary_full.csv")
    
y = dataset.default_payment_next_month.values 
dropList = ["default_payment_next_month"]
dataset.drop(labels=dropList, axis=1, inplace=True)
dataset["default_payment_next_month"] = y

dataset.to_csv("./default_credit_discretized.csv", encoding='utf-8', index=False)
"""