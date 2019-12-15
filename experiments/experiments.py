import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from joblib import Parallel, delayed

from faircorels import load_from_csv, CorelsClassifier
from metrics import ConfusionMatrix, Metric

import csv
import argparse

# parser initialization
parser = argparse.ArgumentParser(description='Evaluation of FairCORELS')
parser.add_argument('--id', type=int, default=1, help='dataset id: 1 for Adult Income, 2 for Compas, 3 for German Credit and 4 for Default Credit')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 - 6')


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
N_ITER = 5*10**6
epsilon_low_regime = np.linspace(0.89, 0.949, num=10) 
epsilon_high_regime = np.linspace(0.95, 0.999, num=60)
epsilon_range = [0.0] + [x for x in epsilon_low_regime] + [x for x in epsilon_high_regime] + [1.0]


# loading dataset

X, y, features, prediction = load_from_csv("../data/{}/{}_rules_full.csv".format(dataset,dataset))

print('nbr features ----------------------->', len(features))

# creating k-folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)
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
                            c=1e-3, 
                            max_card=1, 
                            policy="bfs",
                            bfs_mode=2,
                            mode=3,
                            useUnfairnessLB=True,
                            forbidSensAttr=True,
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

    
    return [acc, unf, acc_train, unf_train,  mdl]

    

# method to run experimer per epsilon and per fairness metric
def per_epsilon(epsilon, fairness_metric, writer):
    
    output = Parallel(n_jobs=-1)(delayed(trainFold)(X_train=fold[0], y_train=fold[1], X_test=fold[2], y_test=fold[3], epsilon=epsilon, fairness_metric=fairness_metric) for fold in folds)

    accuracy = []
    unfairness = []

    accuracy_train = []
    unfairness_train = []

    model = []

    for res in output:
        accuracy.append(res[0])
        unfairness.append(res[1])
        accuracy_train.append(res[2])
        unfairness_train.append(res[3])
        model.append(res[4])

    writer.writerow(
        {'accuracy': np.mean(accuracy),
         'unfairness': np.mean(unfairness),
         'accuracy_train': np.mean(accuracy_train),
         'unfairness_train': np.mean(unfairness_train),
         'epsilon' : epsilon,
         'models' : model
         })
    #print("=========> mean accuracy {}".format(np.mean(accuracy)))
    #print("=========> mean unfairness {}".format(np.mean(unfairness)))


# 1- experiment for statistical_parity
def statistical_parity():
    with open('./results/{}_statistical_parity.csv'.format(dataset), 'a+', newline='') as csvfile:
        fieldnames = ['accuracy', 'unfairness', 'accuracy_train', 'unfairness_train', 'epsilon', 'models']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for eps in epsilon_range:
            print('>>>>>>'*20, eps)
            per_epsilon(eps, 1, writer)


# 2- experiment for predictive_parity
def predictive_parity():
    with open('./results/{}_predictive_parity.csv'.format(dataset), 'a+', newline='') as csvfile:
        fieldnames = ['accuracy', 'unfairness', 'accuracy_train', 'unfairness_train', 'epsilon', 'models']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for eps in epsilon_range:
            print('>>>>>>'*20, eps)
            per_epsilon(eps, 2, writer)


# 3- experiment for predictive_equality
def predictive_equality():
    with open('./results/{}_predictive_equality.csv'.format(dataset), 'a+', newline='') as csvfile:
        fieldnames = ['accuracy', 'unfairness', 'accuracy_train', 'unfairness_train', 'epsilon', 'models']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for eps in epsilon_range:
            print('>>>>>>'*20, eps)
            per_epsilon(eps, 3, writer)


# 4- experiment for equal_opportunity
def equal_opportunity():
    with open('./results/{}_equal_opportunity.csv'.format(dataset), 'a+', newline='') as csvfile:
        fieldnames = ['accuracy', 'unfairness', 'accuracy_train', 'unfairness_train', 'epsilon', 'models']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for eps in epsilon_range:
            print('>>>>>>'*20, eps)
            per_epsilon(eps, 4, writer)


# 5- experiment for conditional_procedure_accuracy_equality (equalized_odds)
def equalized_odds():
    with open('./results/{}_conditional_procedure_accuracy_equality.csv'.format(dataset), 'a+', newline='') as csvfile:
        fieldnames = ['accuracy', 'unfairness', 'accuracy_train', 'unfairness_train', 'epsilon', 'models']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for eps in epsilon_range:
            print('>>>>>>'*20, eps)
            per_epsilon(eps, 5, writer)


# 6- experiment for conditional_use_accuracy_equality
def conditional_use_accuracy_equality():
    with open('./results/{}_conditional_use_accuracy_equality.csv'.format(dataset), 'a+', newline='') as csvfile:
        fieldnames = ['accuracy', 'unfairness', 'accuracy_train', 'unfairness_train', 'epsilon', 'models']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for eps in epsilon_range:
            print('>>>>>>'*20, eps)
            per_epsilon(eps, 6, writer)



experiments = {
    1: statistical_parity,
    2 : predictive_parity,
    3 : predictive_equality,
    4 : equal_opportunity,
    5 : equalized_odds,
    6 : conditional_use_accuracy_equality
}


#run experiment for a particular dataset/fairness metric
experiments[args.metric]()