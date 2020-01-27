import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from joblib import Parallel, delayed, parallel_backend
from faircorels import load_from_csv, CorelsClassifier
from metrics import ConfusionMatrix, Metric
import csv
import argparse
import os
from config import get_data, get_metric, get_strategy
from mpi4py import MPI


# parser initialization
parser = argparse.ArgumentParser(description='Evaluation of FairCORELS')
parser.add_argument('--dataset', type=int, default=1, help='Dataset. 1 Adult, 2: COMPAS, 3: German Credit, 4: Default Credit, 5: Adult_marital, 6: Adult_no_relationship')
parser.add_argument('--metric', type=int, default=1, help='Fairness metric. 1: SP, 2:  PP, 3: PE, 4: EOpp, 5: EOdds, 6: CUAE')
parser.add_argument('--ulb', type=int, default=1, help='Use ULB. 1: Yes, 0: No')
parser.add_argument('--strat', type=int, default=1, help='Search strategy. 1: bfs, 2:curious, 3: lower_bound, 4: bfs_objective_aware')
args = parser.parse_args()


#get dataset and relative infos
dataset, decision, prediction_name, min_feature, min_pos, maj_feature, maj_pos = get_data(args.dataset)

#------------------------setup config

#iterations
#N_ITER = 4*10**6

N_ITER = 10*10**6

#fairness constraint
fairness_metric_name = get_metric(args.metric)
fairness_metric = args.metric

#epsilons
base = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.91, 0.92, 0.93, 0.94]
epsilon_range = base + list(np.linspace(0.95, 0.99, num=20))
epsilons = [round(x,3) for x in epsilon_range] #30 values


# use ulb
ulb = True if args.ulb==1 else False
suffix = "with_ulb" if args.ulb==1 else "without_ulb"


# get search strategy
strategy, bfsMode, strategy_name = get_strategy(args.strat)


#save direcory
save_dir = "./results/{}_{}/{}".format(dataset, suffix, strategy_name)
os.makedirs(save_dir, exist_ok=True)


# load dataset
X, y, features, prediction = load_from_csv("../data/{}/{}_rules_full.csv".format(dataset,dataset))


# creating k-folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)


folds = []
i=0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    folds.append([X_train, y_train, X_test, y_test, i])
    i +=1

cart_product = []
for fold in folds:
    for epsilon in epsilons:
        cart_product.append([fold, epsilon])

#print("------------------------------------------------------->>>>>>>> {}".format(len(cart_product)))

def fit(fold, epsilon, fairness):
    X_train, y_train, X_test, y_test, fold_id = fold[0], fold[1], fold[2], fold[3], fold[4]

    clf = CorelsClassifier(n_iter=N_ITER, 
                            min_support=0.01,
                            c=1e-3, 
                            max_card=1, 
                            policy=strategy,
                            bfs_mode=bfsMode,
                            mode=3,
                            useUnfairnessLB=ulb,
                            forbidSensAttr=True,
                            fairness=fairness, 
                            epsilon=epsilon,
                            maj_pos=maj_pos, 
                            min_pos=min_pos,
                            verbosity=[]
                            )

    clf.fit(X_train, y_train, features=features, prediction_name=prediction_name)

    
    #test
    df_test = pd.DataFrame(X_test, columns=features)
    df_test[decision] = y_test
    df_test["predictions"] = clf.predict(X_test)
    cm_test = ConfusionMatrix(df_test[min_feature], df_test[maj_feature], df_test["predictions"], df_test[decision])
    cm_minority_test, cm_majority_test = cm_test.get_matrix()
    fm_test = Metric(cm_minority_test, cm_majority_test)

    #train 
    df_train = pd.DataFrame(X_train, columns=features)
    df_train[decision] = y_train
    df_train["predictions"] = clf.predict(X_train)
    cm_train = ConfusionMatrix(df_train[min_feature], df_train[maj_feature], df_train["predictions"], df_train[decision])
    cm_minority_train, cm_majority_train = cm_train.get_matrix()
    fm_train = Metric(cm_minority_train, cm_majority_train)

    acc_test = clf.score(X_test, y_test)
    unf_test = fm_test.fairness_metric(fairness_metric)

    acc_train = clf.score(X_train, y_train)
    unf_train = fm_train.fairness_metric(fairness_metric)
    mdl = clf.rl().__str__()

    return [fold_id, epsilon, acc_test, unf_test, acc_train, unf_train, mdl]

def split(container, count):
    return [container[_i::count] for _i in range(count)]

def process_results(epsilons, results, save_path, metric_name):
    # save all fold - eps in dict
    res_dict = {}
    row_list = []
    row_list_model = []

    for res in results:
        res_dict[str(res[0]) + "_" + str(res[1])] = res

    for epsilon in epsilons:
        accuracy_test = []
        unfairness_test = []
        accuracy_train = []
        unfairness_train = []
        row = {}
        row_model = {}

        for fold_id in [0, 1, 2, 3, 4]:
            key = str(fold_id) + "_" + str(epsilon)
            result = res_dict[key]
            acc_test, unf_test, acc_train, unf_train, mdl = result[2], result[3], result[4], result[5], result[6]
            
            accuracy_test.append(acc_test)
            accuracy_train.append(acc_train)

            unfairness_test.append(unf_test)
            unfairness_train.append(unf_train)

            row_model['acc_test_fold_{}'.format(fold_id)] = acc_test
            row_model['acc_train_fold_{}'.format(fold_id)] = acc_train

            row_model['unf_test_fold_{}'.format(fold_id)] = unf_test
            row_model['unf_train_fold_{}'.format(fold_id)] = unf_train

            row_model['mdl_fold_{}'.format(fold_id)] = mdl

        
        row['accuracy'] = np.mean(accuracy_test)
        row['unfairness'] = np.mean(unfairness_test)
        row['accuracy_train'] = np.mean(accuracy_train)
        row['unfairness_train'] = np.mean(unfairness_train)
        row['epsilon'] = np.mean(epsilon)

        row_list.append(row)
        row_list_model.append(row_model)

    filename = '{}/{}.csv'.format(save_path, metric_name)
    filename_model = '{}/{}_model.csv'.format(save_path, metric_name)

    df = pd.DataFrame(row_list)
    df_model = pd.DataFrame(row_list_model)

    df.to_csv(filename, encoding='utf-8', index=False)
    df_model.to_csv(filename_model, encoding='utf-8', index=False)



COMM = MPI.COMM_WORLD

if COMM.rank == 0:
    jobs = split(cart_product, COMM.size)
else:
    jobs = None

jobs = COMM.scatter(jobs, root=0)



results = []
for job in jobs:
    fold = job[0]
    epsilon = job[1]
    #print("----"*20 + ">>> fold: {}, epsilon: {}".format(fold[4], epsilon))
    results.append(fit(fold, epsilon, fairness_metric))


# Gather results on rank 0.
results = MPI.COMM_WORLD.gather(results, root=0)

if COMM.rank == 0:
    results = [_i for temp in results for _i in temp]
    process_results(epsilons, results, save_dir, fairness_metric_name)

