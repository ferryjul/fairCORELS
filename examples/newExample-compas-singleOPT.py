import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from collections import Counter
import argparse
from faircorels import load_from_csv, FairCorelsClassifier, ConfusionMatrix, Metric
import csv
import time

N_ITER = 1*10**7 # The maximum number of nodes in the prefix tree
sensitive_attr_column = 0
unsensitive_attr_column = 1

X, y, features, prediction = load_from_csv("./data/compas_rules_full_single.csv")#("./data/adult_full.csv") # Load the dataset

# parser initialization
parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--epsilon', type=int, default=0, help='epsilon value (min fairness acceptable) for epsilon-constrained method')
parser.add_argument('--filteringMode', type=int, default=0, help='filtering : 0 no, 1 prefix, 2 all extensions')
parser.add_argument('--maxTime', type=int, default=-1, help='filtering : 0 no, 1 prefix, 2 all extensions')
parser.add_argument('--policy', type=str, default="bfs", help='search heuristic - function used to order the priority queue')

#parser.add_argument('--uselb', type=int, default=0, help='use filtering : 0  no, 1  yes')
#parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 statistical_parity, 2 predictive_parity, 3 predictive_equality, 4 equal_opportunity')

args = parser.parse_args()
policy = args.policy

max_time = None
if args.maxTime > 0:
    max_time = args.maxTime

#epsilon_range = np.arange(0.95, 1.001, 0.001) #0.001
#base = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93,0.935, 0.94,0.945]

#epsilon_range = base + list(epsilon_range)
epsilons = [0.7, 0.8, 0.9, 0.95, 0.975, 0.98, 0.985, 0.99, 0.995, 0.999] # 10 values #[round(x,3) for x in epsilon_range] #72 values
#fairnessMetric = args.metric
fairnessMetric = int(np.floor(args.epsilon/len(epsilons))+1)
epsInd = int(args.epsilon - ((fairnessMetric-1)*len(epsilons)))
epsilon = epsilons[epsInd]
if fairnessMetric == 2:
    fairnessMetric = 5
# print("metric=", fairnessMetric, ", epsInd= ", epsInd, "epsilon=", epsilon)
# 10 values for epsilon, 4 fairness metrics, slurm array ranges from 0 to 39
#bools = [False, True]
#useLB = bools[args.uselb]
filteringMode = args.filteringMode
lambdaParam = 1e-3

# We prepare the folds for our 5-folds cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    folds.append([X_train, y_train, X_test, y_test])

accuracy = []
unfairness = []

def compute_unfairness(sensVect, unSensVect, y, y_pred):
    cm = ConfusionMatrix(sensVect, unSensVect, y_pred, y)
    cm_minority, cm_majority = cm.get_matrix()
    fm = Metric(cm_minority, cm_majority)

    if fairnessMetric == 1:
        unf = fm.statistical_parity()
    elif fairnessMetric == 2:
        unf = fm.predictive_parity()
    elif fairnessMetric == 3:
        unf = fm.predictive_equality()
    elif fairnessMetric == 4:
        unf = fm.equal_opportunity()
    elif fairnessMetric == 5:
        unf = fm.equalized_odds()
    elif fairnessMetric == 6:
        unf = fm.conditional_use_accuracy_equality()
    else:
        unf = -1
    
    return unf

def oneFold(foldIndex, X_fold_data): # This part could be multithreaded for better performance
    X_train, y_train, X_test, y_test = X_fold_data

    # Separate protected features to avoid disparate treatment
    # - Training set
    sensVect_train =  X_train[:,sensitive_attr_column]
    unSensVect_train =  X_train[:,unsensitive_attr_column] 
    X_train_unprotected = X_train[:,2:]

    # - Test set
    sensVect_test =  X_test[:,sensitive_attr_column]
    unSensVect_test =  X_test[:,unsensitive_attr_column] 
    X_test_unprotected = X_test[:,2:]

    # Create the FairCorelsClassifier object
    clf = FairCorelsClassifier(n_iter=N_ITER,
                            c=lambdaParam, # sparsity regularization parameter
                            max_card=1, # one rule = one attribute
                            policy=policy, # exploration heuristic
                            bfs_mode=2, # type of BFS: objective-aware
                            mode=3, # epsilon-constrained mode
                            filteringMode=filteringMode,
                            forbidSensAttr=False,
                            map_type="none",
                            fairness=fairnessMetric, 
                            epsilon=epsilon, # fairness constrait
                            verbosity=[], # don't print anything
                            maj_vect=unSensVect_train, # vector defining unprotected group
                            min_vect=sensVect_train, # vector defining protected group
                            min_support = 0.01 
                            )

    start = time.clock()

    # Train it
    clf.fit(X_train_unprotected, y_train, features=features[2:], prediction_name="(recidivism:yes)", time_limit = max_time)# max_evals=100000) # time_limit=8100, 

    time_elapsed = time.clock() - start

    # Print the fitted model
    print("Fold ", foldIndex, " :", clf.rl_, "(RT: ", time_elapsed, " s)")

    # Evaluate our model's accuracy
    accTraining = clf.score(X_train_unprotected, y_train)
    accTest = clf.score(X_test_unprotected, y_test)

    # Evaluate our model's fairness
    train_preds = clf.predict(X_train_unprotected)
    unfTraining = compute_unfairness(sensVect_train, unSensVect_train, y_train, train_preds)

    test_preds = clf.predict(X_test_unprotected)
    unfTest = compute_unfairness(sensVect_test, unSensVect_test, y_test, test_preds)

    # Also compute/collect additional parameters
    length = len(clf.rl_.rules)-1 # -1 because we do not count the default rule
    objF = ((1-accTraining) + (lambdaParam*length)) # best objective function reached
    exploredBeforeBest = int(clf.nbExplored)
    cacheSizeAtExit = int(clf.nbCache)
    return [foldIndex, accTraining, unfTraining, objF, accTest, unfTest, exploredBeforeBest, cacheSizeAtExit, length, time_elapsed,  clf.get_solving_status()]

# Run training/evaluation for all folds using multi-threading
ret = Parallel(n_jobs=-1)(delayed(oneFold)(foldIndex, X_fold_data) for foldIndex, X_fold_data in enumerate(folds))

# Unwrap the results
accuracy = [ret[i][4] for i in range(0,5)]
unfairness = [ret[i][5] for i in range(0,5)]
objective_functions = [ret[i][3] for i in range(0,5)]
accuracyT = [ret[i][1] for i in range(0,5)]
unfairnessT = [ret[i][2] for i in range(0,5)]

print("=========> Training Accuracy (average)= ", np.average(accuracyT))
print("=========> Training Unfairness (average)= ", np.average(unfairnessT))
print("=========> Training Objective function value (average)= ", np.average(objective_functions))
print("=========> Test Accuracy (average)= ", np.average(accuracy))
print("=========> Test Unfairness (average)= ", np.average(unfairness))

#Save results in a csv file
resPerFold = dict()
for aRes in ret:
    resPerFold[aRes[0]] = [aRes[1], aRes[2], aRes[3], aRes[4], aRes[5], aRes[6], aRes[7], aRes[8], aRes[9], aRes[10]]

if max_time is None: 
    fileName = './results/faircorels_eps%f_metric%d_LB%d_%s_single.csv' %(epsilon, fairnessMetric, filteringMode, policy)
else:
    fileName = './results/faircorels_eps%f_metric%d_LB%d_%s_tLimit%d_single.csv' %(epsilon, fairnessMetric, filteringMode, policy, max_time)
with open(fileName, mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Fold#', 'Training accuracy', 'Training Unfairness(%d)' %fairnessMetric, 'Training objective function', 'Test accuracy', 'Test unfairness', '#Nodes explored for best solution', 'Cache size for best solution', 'Average length', 'CPU running time (s)', 'Solving Status'])#, 'Fairness STD', 'Accuracy STD'])
    for index in range(5):
        csv_writer.writerow([index, resPerFold[index][0], resPerFold[index][1], resPerFold[index][2], resPerFold[index][3], resPerFold[index][4], resPerFold[index][5], resPerFold[index][6], resPerFold[index][7], resPerFold[index][8], resPerFold[index][9]])#,fairnessSTD_list_test_tot[i][fairnessMetric-1], accuracySTD_list_test_tot[i]])
    csv_writer.writerow(['Average', np.average(accuracyT), np.average(unfairnessT), np.average(objective_functions), np.average(accuracy), np.average(unfairness), '', ''])#,fairnessSTD_list_test_tot[i][fairnessMetric-1], accuracySTD_list_test_tot[i]])
