import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from collections import Counter

from faircorels import load_from_csv, CorelsClassifier
from metrics import ConfusionMatrix, Metric

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

if args.id==8:
    dataset = "adult_perfs"
    decision = "income"
    prediction_name="[income:>50K]"
    min_feature = "gender_Female"
    min_pos = 1
    maj_feature = "gender_Male"
    maj_pos = 2

# use sens. attri
forbidSensAttr = True if args.attr==1 else False
suffix = "without_dem" if args.attr==1 else "with_dem"

#save direcory
save_dir = "./perfs_results/{}_{}".format(dataset,suffix)
os.makedirs(save_dir, exist_ok=True)


# parameters
N_ITER = 10*10**6
fairness_metric = 1
epsilon = 0.95
reg = 0.0 #1e-3
min_support = 0.01
useUnfairnessLB = True



X, y, features, prediction = load_from_csv("../data/{}/{}_rules_full.csv".format(dataset,dataset))


kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy = []
unfairness = []

folds = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    folds.append([X_train, y_train, X_test, y_test])
    


def trainFold(X_train, y_train, X_test, y_test):

    clf = CorelsClassifier(n_iter=N_ITER, 
                            min_support=min_support,
                            c=reg, 
                            max_card=1, 
                            policy="bfs",
                            bfs_mode=2,
                            mode=3,
                            useUnfairnessLB=useUnfairnessLB,
                            forbidSensAttr=forbidSensAttr,
                            fairness=fairness_metric, 
                            epsilon=epsilon,
                            maj_pos=maj_pos, 
                            min_pos=min_pos,
                            verbosity=["rulelist"]
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

    
    return [acc, unf, acc_train, unf_train]



output = Parallel(n_jobs=-1)(delayed(trainFold)(X_train=fold[0], y_train=fold[1], X_test=fold[2], y_test=fold[3]) for fold in folds)

accuracy = []
unfairness = []
accuracy_train = []
unfairness_train = []

for res in output:
    accuracy.append(res[0])
    unfairness.append(res[1])
    accuracy_train.append(res[2])
    unfairness_train.append(res[3])


print("=========>  accuracy test {}".format(np.mean(accuracy)))
print("=========>  unfairness test {}".format(np.mean(unfairness)))
print('----'*20)
print("=========>  accuracy train {}".format(np.mean(accuracy_train)))
print("=========>  unfairness train {}".format(np.mean(unfairness_train)))


row = {
    'accuracy': np.mean(accuracy),
    'unfairness': np.mean(unfairness),
    'accuracy_train': np.mean(accuracy_train),
    'unfairness_train': np.mean(unfairness_train)
}


filename = '{}/{}.csv'.format(save_dir, metrics[args.metric])
df = pd.DataFrame([row])
df.to_csv(filename, encoding='utf-8', index=False)
