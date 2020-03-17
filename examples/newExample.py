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



sensVect =  X_train[:,32]
unSensVect =  X_train[:,33]
print("Sensitive/Unsensitive attributes vectors shapes : ", sensVect.shape)
print("Labels vector shape : ", y_train.shape)
unique, counts = np.unique(sensVect, return_counts=True)
sensDict = dict(zip(unique, counts))
print("Sensitive vector captures %d instances" %sensDict[1])
unique2, counts2 = np.unique(unSensVect, return_counts=True)
unSensDict = dict(zip(unique2, counts2))
print("Unsensitive vector captures %d instances" %unSensDict[1])
clf = CorelsClassifier(n_iter=N_ITER, 
                        c=1e-3, 
                        max_card=2, 
                        policy="bfs",
                        bfs_mode=2,
                        mode=3,
                        useUnfairnessLB=True,
                        forbidSensAttr=False,
                        fairness=fairnessMetric, 
                        epsilon=0.9,
                        verbosity=[],
                        maj_vect=unSensVect,
                        min_vect=sensVect,
                        min_support = 0.01
                        )
clf.fit(X_train, y_train, features=features, prediction_name="(income:>50K)")
df_test = pd.DataFrame(X_test, columns=features)
df_test['income'] = y_test
df_test["predictions"] = clf.predict(X_test)
print(clf.predict(X_test))
print("predict with scores :")
print(clf.predict_with_scores(X_test))
cm = ConfusionMatrix(df_test["gender_Female"], df_test["gender_Male"], df_test["predictions"], df_test["income"])
cm_minority, cm_majority = cm.get_matrix()
fm = Metric(cm_minority, cm_majority)

acc = clf.score(X_test, y_test)
unf = fm.statistical_parity()
length = len(clf.rl_.rules)
print(clf.rl_)
print("=========> accuracy {}".format(acc))
print("=========> unfairness {}".format(unf))