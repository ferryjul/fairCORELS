import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
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
    dataset = "german_credit_v2"
    decision = "credit_rating"
    prediction_name="[good_credit_rating]"
    min_feature = "age:<25"
    min_pos = 1
    maj_feature = "age:>=25"
    maj_pos = 2

# parameters
N_ITER = 10*10**6


# epsilon range
epsilon_range = np.arange(0.95, 1.001, 0.001)
base = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
epsilon_range = base + list(epsilon_range)
epsilon_range = [round(x,3) for x in epsilon_range] #60 values

n_eps = 30 


# use sens. attri
forbidSensAttr = True if args.attr==1 else False
suffix = "without_dem" if args.attr==1 else "with_dem"

#save direcory
save_dir = "./results/{}_{}".format(dataset,suffix)
os.makedirs(save_dir, exist_ok=True)

# loading dataset

X, y, features, prediction = load_from_csv("../data/{}/{}_rules_full.csv".format(dataset,dataset))



# creating k-folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy = []
unfairness = []

folds = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    folds.append([X_train, y_train, X_test, y_test])



# method to run each folds
def trainFold(X_train, y_train, X_test, y_test, epsilon, fairness_metric):

    clf = CorelsClassifier(n_iter=N_ITER, 
                            min_support=0.01,
                            c=1e-8, 
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
    mdl = {'accuracy': acc, 'unfairness':unf, 'accuracy_train': acc_train, 'unfairness_train':unf_train, 'description': clf.rl().__str__()}

    return [acc, unf, acc_train, unf_train,  mdl, epsilon]
    

# method to run experimer per epsilon and per fairness metric
def per_fold(fold, epsilons, fairness_metric):
    
    output = Parallel(n_jobs=n_eps)(delayed(trainFold)(
                                                X_train=fold[0], 
                                                y_train=fold[1], 
                                                X_test=fold[2], 
                                                y_test=fold[3], 
                                                epsilon=epsilon, 
                                                fairness_metric=fairness_metric) for epsilon in epsilons)

    accuracy = []
    unfairness = []
    accuracy_train = []
    unfairness_train = []
    model = []
    eps = []

    for res in output:
        accuracy.append(res[0])
        unfairness.append(res[1])
        accuracy_train.append(res[2])
        unfairness_train.append(res[3])
        model.append(res[4])
        eps.append(res[5])

    cols = {
            'accuracy': accuracy,
            'unfairness': unfairness,
            'accuracy_train': accuracy_train,
            'unfairness_train': unfairness_train,
            'model' : model,
         }

    

    return cols
    


def run():
    filename = '{}/{}.csv'.format(save_dir, metrics[args.metric])
    rowlist = []
        
    cols = [per_fold(fold=fold, epsilons=epsilon_range, fairness_metric=args.metric) for fold in folds]

    for idx, eps in enumerate(epsilon_range):
        accuracy = []
        unfairness = []
        accuracy_train = []
        unfairness_train = []
        model = []

        for col in cols:
            accuracy.append(col['accuracy'][idx])
            unfairness.append(col['unfairness'][idx])
            accuracy_train.append(col['accuracy_train'][idx])
            unfairness_train.append(col['unfairness_train'][idx])
            model.append(col['model'][idx])

        row = { 'accuracy': np.mean(accuracy),
                'unfairness': np.mean(unfairness),
                'accuracy_train': np.mean(accuracy_train),
                'unfairness_train': np.mean(unfairness_train),
                'epsilon' : eps,
                'models' : model
               }

        rowlist.append(row)

    df = pd.DataFrame(rowlist)
    df.to_csv(filename, encoding='utf-8', index=False)


run()