import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, train_test_split
from joblib import Parallel, delayed, parallel_backend

from faircorels import load_from_csv, CorelsClassifier
from metrics import ConfusionMatrix, Metric

import csv
import argparse

import os


# parser initialization
parser = argparse.ArgumentParser(description='Evaluation of FairCORELS')
parser.add_argument('--id', type=int, default=1, help='dataset id: 1 for Adult Income, 2 for Compas, 3 for German Credit and 4 for Default Credit')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 - 6')
parser.add_argument('--attr', type=int, default=1, help='use sensitive attribute: 1 no, 2 yes')



args = parser.parse_args()

# metrics

metrics = {
    1: "statistical_parity",
    2 : "predictive_parity",
    3 : "predictive_equality",
    4 : "equal_opportunity",
    5 : "equalized_odds",
    6 : "conditional_use_accuracy_equality"
}


# dataset details
dataset, decision, prediction_name, min_feature, min_pos, maj_feature, maj_pos = None, None, None, None, None, None, None

if args.id==1:
    dataset = "adult"
    decision = "income"
    prediction_name="[income:>50K]"
    min_feature = "gender_Female"
    min_pos = 1
    maj_feature = "gender_Male"
    maj_pos = 2

if args.id==2:
    dataset = "compas"
    decision = "two_year_recid"
    prediction_name="[two_year_recid]"
    min_feature = "race_African-American"
    min_pos = 1
    maj_feature = "race_Caucasian"
    maj_pos = 2

if args.id==3:
    dataset = "german_credit"
    decision = "credit_rating"
    prediction_name="[good_credit_rating]"
    min_feature = "age:<25"
    min_pos = 1
    maj_feature = "age:>=25"
    maj_pos = 2

if args.id==4:
    dataset = "default_credit"
    decision = "default_payment_next_month"
    prediction_name="[default_payment_next_month]"
    min_feature = "gender:Female"
    min_pos = 1
    maj_feature = "gender:Male"
    maj_pos = 2

if args.id==5:
    dataset = "adult_marital"
    decision = "income"
    prediction_name="[income:>50K]"
    min_feature = "maritalStatus_single"
    min_pos = 1
    maj_feature = "maritalStatus_married"
    maj_pos = 2

if args.id==6:
    dataset = "adult_no_relationship"
    decision = "income"
    prediction_name="[income:>50K]"
    min_feature = "gender_Female"
    min_pos = 1
    maj_feature = "gender_Male"
    maj_pos = 2


if args.id==7:
    dataset = "adult_no_relationship_neg"
    decision = "income"
    prediction_name="[income:>50K]"
    min_feature = "gender_Female"
    min_pos = 1
    maj_feature = "gender_Male"
    maj_pos = 2

if args.id==8:
    dataset = "compas_neg"
    decision = "two_year_recid"
    prediction_name="[two_year_recid]"
    min_feature = "race_African-American"
    min_pos = 1
    maj_feature = "race_Caucasian"
    maj_pos = 2



# parameters
N_ITER = 10*10**6

# fairness metric / epsilon range
fairness_metric = 1
epsilon = 0.998


# use sens. attri
forbidSensAttr = True if args.attr==1 else False



# loading dataset

X, y, features, prediction = load_from_csv("../data/{}/{}_rules_full.csv".format(dataset,dataset))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


clf = CorelsClassifier(n_iter=N_ITER, 
                        min_support=0.01,
                        c=1e-3, 
                        max_card=1, 
                        policy="bfs",
                        bfs_mode=2,
                        mode=3,
                        useUnfairnessLB=False,
                        forbidSensAttr=forbidSensAttr,
                        fairness=fairness_metric, 
                        epsilon=epsilon,
                        maj_pos=maj_pos, 
                        min_pos=min_pos,
                        verbosity=["progress", "rulelist"]
                        )

clf.fit(X_train, y_train, features=features, prediction_name=prediction_name)

    




df_test = pd.DataFrame(X_test, columns=features)
df_test[decision] = y_test
df_test["predictions"] = clf.predict(X_test)
cm = ConfusionMatrix(df_test[min_feature], df_test[maj_feature], df_test["predictions"], df_test[decision])
cm_minority, cm_majority = cm.get_matrix()
fm = Metric(cm_minority, cm_majority)

acc = clf.score(X_test, y_test)
unf = fm.fairness_metric(fairness_metric)
print('-------------------------- accuracy: {}'.format(acc))
print('-------------------------- unfairness: {}'.format(unf))

    
