import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, train_test_split

from joblib import Parallel, delayed
from collections import Counter

from faircorels import load_from_csv, CorelsClassifier
from metrics import ConfusionMatrix, Metric

import argparse

# parser initialization
parser = argparse.ArgumentParser(description='Evaluation of FairCORELS')
parser.add_argument('--id', type=int, default=1, help='dataset id: 1 for Adult Income, 2 for Compas, 3 for German Credit and 4 for Default Credit')

args = parser.parse_args()


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



# parameters
N_ITER = 2*10**6
fairness_metric = 3
epsilon = 0.99
_lambda = 1e-3
forbidSensAttr = True



X, y, features, prediction = load_from_csv("../data/{}/{}_rules_full.csv".format(dataset, dataset))

print('nbr features ----------------------->', len(features))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print('train', len(X_train))
print('test', len(X_test))


def trainFold(X_train, y_train, X_test, y_test):

    clf = CorelsClassifier(n_iter=N_ITER, 
                            min_support=0.01,
                            c=_lambda, 
                            max_card=1, 
                            policy="bfs",
                            bfs_mode=2,
                            mode=3,
                            useUnfairnessLB=True,
                            forbidSensAttr=forbidSensAttr,
                            fairness=fairness_metric, 
                            epsilon=epsilon,
                            maj_pos=maj_pos, 
                            min_pos=min_pos,
                            verbosity=["loud", "rulelist"]
                            )


    clf.fit(X_train, y_train, features=features, prediction_name=prediction_name)
    
    #test
    df_test = pd.DataFrame(X_test, columns=features)
    df_test[decision] = y_test
    df_test["predictions"] = clf.predict(X_test)
    cm = ConfusionMatrix(df_test[min_feature], df_test[maj_feature], df_test["predictions"], df_test[decision])
    cm_minority, cm_majority = cm.get_matrix()
    fm = Metric(cm_minority, cm_majority)

    #train 
    df_train = pd.DataFrame(X_train, columns=features)
    df_train[decision] = y_train
    df_train["predictions"] = clf.predict(X_train)
    cm_train = ConfusionMatrix(df_train[min_feature], df_train[maj_feature], df_train["predictions"], df_train[decision])
    cm_minority_train, cm_majority_train = cm_train.get_matrix()
    fm_train = Metric(cm_minority_train, cm_majority_train)

    acc = clf.score(X_test, y_test)
    unf = fm.fairness_metric(fairness_metric)

    acc_train = clf.score(X_train, y_train)
    unf_train = fm_train.fairness_metric(fairness_metric)

    
    print("=========>  accuracy test {}".format(acc))
    print("=========>  unfairness test {}".format(unf))

    print("=========>  accuracy train {}".format(acc_train))
    print("=========>  unfairness train {}".format(unf_train))


trainFold(X_train, y_train, X_test, y_test)