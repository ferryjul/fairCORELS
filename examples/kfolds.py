import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from collections import Counter

from faircorels import load_from_csv, CorelsClassifier
from metrics import ConfusionMatrix, Metric
import csv
import time

N_ITER = 1*10**5

X, y, features, prediction = load_from_csv("./data/adult_full.csv")

fairnessMetric = 1

kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy = []
unfairness = []

folds = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    folds.append([X_train, y_train, X_test, y_test])
    print('------------------------------>>>>>>>>>>')
    print('train distribution ------', sorted(Counter(y_train).items()))
    print('test distribution ------', sorted(Counter(y_test).items()))



def trainFold(X_train, y_train, X_test, y_test, min_supp):
    sensVect =  [row[32] for row in X_train]
    unSensVect =  [row[33] for row in X_train]
    print("Sensitive attributes vector : captures ", sensVect.count(1) ,"/", len(sensVect), " instances", "Unsensitive attributes vector : captures ", unSensVect.count(1) ,"/", len(unSensVect), " instances")
    
    clf = CorelsClassifier(n_iter=N_ITER, 
                            c=1e-3, 
                            max_card=1, 
                            policy="bfs",
                            bfs_mode=2,
                            mode=3,
                            useUnfairnessLB=True,
                            forbidSensAttr=False,
                            fairness=fairnessMetric, 
                            epsilon=0.9,
                            verbosity=[],
                            #maj_vect=unSensVect,
                            min_pos=32,
                            min_support = min_supp
                            )
    clf.fit(X_train, y_train, features=features, prediction_name="(income:>50K)")
    df_test = pd.DataFrame(X_test, columns=features)
    df_test['income'] = y_test
    df_test["predictions"] = clf.predict(X_test)
    cm = ConfusionMatrix(df_test["gender_Female"], df_test["gender_Male"], df_test["predictions"], df_test["income"])
    cm_minority, cm_majority = cm.get_matrix()
    fm = Metric(cm_minority, cm_majority)

    acc = clf.score(X_test, y_test)
    unf = fm.statistical_parity()
    length = len(clf.rl_.rules)
    #print("=========> accuracy {}".format(acc))
    #print("=========> unfairness {}".format(unf))

    return [acc, unf, length]

mList = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
meanAcc = []
meanUnf = []
medianAcc = []
medianUnf = []
times = []
meanLength = []

for m in mList:
    start_time = time.time()
    output = Parallel(n_jobs=-1)(delayed(trainFold)(X_train=fold[0], y_train=fold[1], X_test=fold[2], y_test=fold[3], min_supp=m) for fold in folds)
    accuracy = []
    unfairness = []
    length = []
    for res in output:
        accuracy.append(res[0])
        unfairness.append(res[1])
        length.append(res[2])
    print("-------------- min_support = ", m, " ------------")
    print("=========> median accuracy {}".format(np.median(accuracy)))
    print("=========> median unfairness {}".format(np.median(unfairness)))

    print("=========> mean accuracy {}".format(np.mean(accuracy)))
    print("=========> mean unfairness {}".format(np.mean(unfairness)))
    meanLength.append(np.mean(length))
    meanAcc.append(np.mean(accuracy))
    meanUnf.append(np.mean(unfairness))
    medianAcc.append(np.median(accuracy))
    medianUnf.append(np.median(unfairness))
    times.append(time.time() - start_time)

name_csv = './results_min_support_new.csv'
with open(name_csv, mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Min_support', 'Mean accuracy', 'Mean unfairness(%d)' %fairnessMetric, 'Median accuracy', 'Median unfairness(%d)' %fairnessMetric, 'Running time', 'Mean length'])
    index = 0
    for i in range(len(mList)):
        csv_writer.writerow([mList[i], meanAcc[i], meanUnf[i], medianAcc[i], medianUnf[i], times[i], meanLength[i]])
    index = index + 1
    csv_writer.writerow(["", "", "", "", ""])
print("csv file saved : %s" %name_csv)