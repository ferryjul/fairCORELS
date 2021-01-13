import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from collections import Counter
import argparse
from faircorels import load_from_csv, CorelsClassifier
from metrics import ConfusionMatrix, Metric
import csv
import time


N_ITER = 2*10**5 # The maximum number of nodes in the prefix tree
sensitive_attr_column = 0
unsensitive_attr_column = 1

X, y, features, prediction = load_from_csv("./data/compas_rules_full.csv")#("./data/adult_full.csv") # Load the dataset

print("Sensitive attribute is ", features[sensitive_attr_column])
print("Unsensitive attribute is ", features[unsensitive_attr_column])
# The type of fairness metric used. 
# -> 1 : statistical parity, 2 : predictive parity, 3 : predictive equality, 4 : equal opportunity, 5 : equalized odds, 6 : conditional use accuracy equality

# parser initialization
parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--epsilon', type=int, default=0, help='epsilon value (min fairness acceptable) for epsilon-constrained method')
parser.add_argument('--uselb', type=int, default=0, help='use filtering : 0  no, 1  yes')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 statistical_parity, 2 predictive_parity, 3 predictive_equality, 4 equal_opportunity, 5 Equalized Odds, 6 Conditional use accuracy equality')

args = parser.parse_args()

# list of values for epsilon, we take the given index
epsilon_range = np.arange(0.95, 1.001, 0.002) #0.001
base = [0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93,0.935, 0.94,0.945]
epsilon_range = base + list(epsilon_range)
epsilons = [round(x,3) for x in epsilon_range] #60 values
epsilon = epsilons[args.epsilon]
print("epsilons = ", epsilons, ", len = ", len(epsilons))
print("epsilon = ", epsilon)

fairnessMetric = args.metric
bools = [False, True]
useLB = bools[args.uselb]
lambdaParam = 1e-3
# We prepare the folds for our 5-folds cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    folds.append([X_train, y_train, X_test, y_test])
    print('------------------------------>>>>>>>>>>')
    print('train distribution ------', sorted(Counter(y_train).items()))
    print('test distribution ------', sorted(Counter(y_test).items()))

accuracy = []
unfairness = []

def oneFold(foldIndex, X_fold_data): # This part could be multithreaded for better performance
    print("Fold ", foldIndex)
    X_train, y_train, X_test, y_test = X_fold_data

    # Prepare the vectors defining the protected (sensitive) and unprotected (unsensitive) groups
    # Uncomment the print to get information about the sensitive/unsensitive vectors
    # We will then remove the first two columns (corresp. to sens and unsens attrs here) from training data
    # to prevent disparate treatment
    sensVect =  X_train[:,sensitive_attr_column]
    unSensVect =  X_train[:,unsensitive_attr_column] 

    unique, counts = np.unique(sensVect, return_counts=True)
    sensDict = dict(zip(unique, counts))
    print("Sensitive vector captures %d instances" %sensDict[1])
    unique2, counts2 = np.unique(unSensVect, return_counts=True)
    unSensDict = dict(zip(unique2, counts2))
    print("Unsensitive vector captures %d instances" %unSensDict[1])

    # Create the CorelsClassifier object
    clf = CorelsClassifier(n_iter=N_ITER, 
                            c=lambdaParam, 
                            max_card=1, 
                            policy="bfs",
                            bfs_mode=2,
                            mode=3,
                            useUnfairnessLB=useLB,
                            forbidSensAttr=False,
                            fairness=fairnessMetric, 
                            epsilon=epsilon, 
                            verbosity=["rulelist"],
                            maj_vect=unSensVect, 
                            min_vect=sensVect, 
                            min_support = 0.01 
                            )
    # Train it
    clf.fit(X_train[:,2:], y_train, features=features[2:], prediction_name="(recidivism:yes)")#, time_limit = 10)# max_evals=100000) # time_limit=8100, 

    # Prepare the test set
    df_test = pd.DataFrame(X_test, columns=features)
    df_test['two_year_recid'] = y_test

    # Compute our model's predictions
    df_test["predictions"] = clf.predict(X_test[:,2:])
    cm = ConfusionMatrix(df_test["race_African-American"], df_test["race_Caucasian"], df_test["predictions"], df_test["two_year_recid"])
    cm_minority, cm_majority = cm.get_matrix()
    fm = Metric(cm_minority, cm_majority)

    acc = clf.score(X_test[:,2:], y_test)
    if fairnessMetric == 1:
        unf = fm.statistical_parity()
    elif fairnessMetric == 2:
        unf = fm.predictive_parity()
    elif fairnessMetric == 3:
        unf = fm.predictive_equality()
    elif fairnessMetric == 4:
        unf = fm.equal_opportunity()
    else:
        unf = -1
    length = len(clf.rl_.rules)-1

    accTraining = clf.score(X_train[:,2:], y_train)
    
    # Prepare the training set eval
    df_train = pd.DataFrame(X_train, columns=features)
    df_train['two_year_recid'] = y_train

    # Compute our model's predictions
    df_train["predictions"] = clf.predict(X_train[:,2:])
    cm_train = ConfusionMatrix(df_train["race_African-American"], df_train["race_Caucasian"], df_train["predictions"], df_train["two_year_recid"])
    cm_minority_train, cm_majority_train = cm_train.get_matrix()
    fm_train = Metric(cm_minority_train, cm_majority_train)

    if fairnessMetric == 1:
        unfTraining = fm_train.statistical_parity()
    elif fairnessMetric == 2:
        unfTraining = fm_train.predictive_parity()
    elif fairnessMetric == 3:
        unfTraining = fm_train.predictive_equality()
    elif fairnessMetric == 4:
        unfTraining = fm_train.equal_opportunity()
    else:
        unfTraining = -1
    #print("Fold ", foldIndex, " computed rule list : \n", clf.rl_)
    #print("Fold ", foldIndex, " accuracy ", acc)
    #print("Fold ", foldIndex, " unfairness ", unf)
    objF = ((1-accTraining) + (lambdaParam*length))
    #print("Fold ", foldIndex, " objective function value (training set) : ", objF)
    #print("Fold ", foldIndex, " #explored stats : ", int(clf.nbExplored), "#cache : ", int(clf.nbCache) )

    return [foldIndex, accTraining, unfTraining, objF, acc, unf, int(clf.nbExplored), int(clf.nbCache), length]

ret = Parallel(n_jobs=-1)(delayed(oneFold)(foldIndex, X_fold_data) for foldIndex, X_fold_data in enumerate(folds))
accuracy = [ret[i][4] for i in range(0,5)]
unfairness = [ret[i][5] for i in range(0,5)]
objective_functions = [ret[i][3] for i in range(0,5)]
accuracyT = [ret[i][1] for i in range(0,5)]
unfairnessT = [ret[i][2] for i in range(0,5)]
print("=========> Accuracy (average)= ", np.average(accuracy))
print("=========> Unfairness (average)= ", np.average(unfairness))
print("=========> Objective function value (average)= ", np.average(objective_functions))

resPerFold = dict()
for aRes in ret:
    resPerFold[aRes[0]] = [aRes[1], aRes[2], aRes[3], aRes[4], aRes[5], aRes[6], aRes[7], aRes[8]]

# save results
with open('./results/faircorels_eps%f_metric%d_LB%d.csv' %(epsilon, fairnessMetric, useLB), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Fold#', 'Training accuracy', 'Training Unfairness(%d)' %fairnessMetric, 'Training objective function', 'Test accuracy', 'Test unfairness', '#Nodes explored for best solution', 'Cache size for best solution', 'Average length'])#, 'Fairness STD', 'Accuracy STD'])
    for index in range(5):
        csv_writer.writerow([index, resPerFold[index][0], resPerFold[index][1], resPerFold[index][2], resPerFold[index][3], resPerFold[index][4], resPerFold[index][5], resPerFold[index][6], resPerFold[index][7]])#,fairnessSTD_list_test_tot[i][fairnessMetric-1], accuracySTD_list_test_tot[i]])
    csv_writer.writerow(['Average', np.average(accuracyT), np.average(unfairnessT), np.average(objective_functions), np.average(accuracy), np.average(unfairness), '', ''])#,fairnessSTD_list_test_tot[i][fairnessMetric-1], accuracySTD_list_test_tot[i]])
    csv_writer.writerow(["", "", "", ""])#, "", ""])
