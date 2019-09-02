import numpy as np
import pandas as pd
from faircorels import *
from metrics import ConfusionMatrix, Metric
from sklearn.model_selection import KFold
import csv
import os.path
from sklearn.externals.joblib import Parallel, delayed

###------------->> Loading data

df = pd.read_csv("data/adult_full.csv")

y = df['income']
df.drop(labels=['income'], axis=1, inplace=True)
features = list(df)
X = df

epsilons = [0.95, 0.96, 0.97, 0.98, 0.99]
betas = [0.1, 0.2, 0.3, 0.4, 0.5]
iterations = 1000000



###------------->> epsilon constraint
def epsilon_fold(train, test, eps=0.95, metric=1, n=1000000):
    print("=========================> epsilon: {}, metric: {}".format(eps, metric))
    # prepare input
    X_train = X.iloc[train, :]
    y_train = y.iloc[train]
    X_test = X.iloc[test, :]
    y_test = y.iloc[test]

    #init clf
    clf = CorelsClassifier(n_iter=n, 
                            c=0.0005, 
                            max_card=1, 
                            policy="bfs", 
                            bfs_mode=2, 
                            useUnfairnessLB=True, 
                            fairness=metric, 
                            min_pos=19, 
                            maj_pos=20, 
                            epsilon=eps, 
                            mode=3, 
                            verbosity=[])
    
    #fit clf
    clf.fit(X_train, y_train, features=features, prediction_name="income")

    # predictions and accuracy on the test set
    predictions_test = clf.predict(X_test)
    acc_test = clf.score(X_test, y_test)

    # fairness and accuracy on the test set
    cm_test = ConfusionMatrix(X_test["gender:Female"], X_test["gender:Male"], predictions_test, y_test)
    cm_minority_test, cm_majority_test = cm_test.get_matrix()
    fm_test = Metric(cm_minority_test, cm_majority_test)

    unfairness = 0.0

    if metric == 1:
        unfairness = fm_test.statistical_parity()
    if metric == 2:
        unfairness = fm_test.predictive_parity()
    if metric == 3:
        unfairness = fm_test.predictive_equality()
    if metric == 4:
        unfairness = fm_test.equal_opportunity()
    
    return [acc_test, unfairness]

###------------->> beta regularization
def beta_fold(train, test, beta=0.1, metric=1, n=1000000):
    print("=========================> beta: {}, metric: {}".format(beta, metric))
    # prepare input
    X_train = X.iloc[train, :]
    y_train = y.iloc[train]
    X_test = X.iloc[test, :]
    y_test = y.iloc[test]

    #init clf
    clf = CorelsClassifier(n_iter=n, 
                            c=0.0005, 
                            max_card=1, 
                            policy="bfs", 
                            bfs_mode=2, 
                            useUnfairnessLB=True, 
                            fairness=metric, 
                            min_pos=19, 
                            maj_pos=20, 
                            beta=beta, 
                            mode=1, 
                            verbosity=[])
    
    #fit clf
    clf.fit(X_train, y_train, features=features, prediction_name="income")

    # predictions and accuracy on the test set
    predictions_test = clf.predict(X_test)
    acc_test = clf.score(X_test, y_test)

    # fairness and accuracy on the test set
    cm_test = ConfusionMatrix(X_test["gender:Female"], X_test["gender:Male"], predictions_test, y_test)
    cm_minority_test, cm_majority_test = cm_test.get_matrix()
    fm_test = Metric(cm_minority_test, cm_majority_test)

    unfairness = 0.0

    if metric == 1:
        unfairness = fm_test.statistical_parity()
    if metric == 2:
        unfairness = fm_test.predictive_parity()
    if metric == 3:
        unfairness = fm_test.predictive_equality()
    if metric == 4:
        unfairness = fm_test.equal_opportunity()
    
    return [acc_test, unfairness]




kf = KFold(n_splits=10)


for m in [1, 2, 3, 4]:
    data = {
    'parameter': [],
    'fold_id': [],
    'accuracy': [],
    'unfairness': []
    }

    for eps in epsilons:
        results = Parallel(n_jobs=-1)(delayed(epsilon_fold)(train, test, eps=eps, metric=m, n=iterations) for train, test in kf.split(X))
        for idx, res in enumerate(results):
            data['parameter'].append(eps)
            data['fold_id'].append(idx)
            data['accuracy'].append(res[0])
            data['unfairness'].append(res[1])
    df = pd.DataFrame(data)
    filename = './output/df_epsilon_fairness_{}.csv'.format(m)
    df.to_csv(filename, encoding='utf-8', index=False)


for m in [1, 2, 3, 4]:
    data = {
    'parameter': [],
    'fold_id': [],
    'accuracy': [],
    'unfairness': []
    }

    for beta in betas:
        results = Parallel(n_jobs=-1)(delayed(beta_fold)(train, test, beta=beta, metric=m, n=iterations) for train, test in kf.split(X))
        for idx, res in enumerate(results):
            data['parameter'].append(beta)
            data['fold_id'].append(idx)
            data['accuracy'].append(res[0])
            data['unfairness'].append(res[1])
    df = pd.DataFrame(data)
    filename = './output/df_beta_fairness_{}.csv'.format(m)
    df.to_csv(filename, encoding='utf-8', index=False)

            
        


