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

N_ITER = 1*10**5 # The maximum number of nodes in the prefix tree

X, y, features, prediction = load_from_csv("./data/adult_full.csv") # Load the dataset

fairnessMetric = 1 # 1 = Statistical parity

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

for foldIndex, X_fold_data in enumerate(folds): # This part could be multithreaded for better performance
    X_train, y_train, X_test, y_test = X_fold_data

    # Prepare the vectors defining the protected (sensitive) and unprotected (unsensitive) groups
    # Uncomment the print to get information about the sensitive/unsensitive vectors
    sensVect =  X_train[:,32] # 32 = female 
    unSensVect =  X_train[:,33] # 33 = male (note that here we could directly give the column number)
    #print("Sensitive/Unsensitive attributes vectors shapes : ", sensVect.shape)
    #print("Labels vector shape : ", y_train.shape)
    unique, counts = np.unique(sensVect, return_counts=True)
    sensDict = dict(zip(unique, counts))
    #print("Sensitive vector captures %d instances" %sensDict[1])
    unique2, counts2 = np.unique(unSensVect, return_counts=True)
    unSensDict = dict(zip(unique2, counts2))
    #print("Unsensitive vector captures %d instances" %unSensDict[1])

    # Create the CorelsClassifier object
    clf = CorelsClassifier(n_iter=N_ITER, 
                            c=1e-3, 
                            max_card=2, 
                            policy="bfs",
                            bfs_mode=2,
                            mode=3,
                            useUnfairnessLB=True,
                            forbidSensAttr=False,
                            fairness=fairnessMetric, 
                            epsilon=0.99,
                            verbosity=[],
                            maj_vect=unSensVect, # In our example, maj_pos = 33 would give the same result (not specifying unprotected group would give the same result as well)
                            min_vect=sensVect, # In our example, min_pos = 32 would give the same result
                            min_support = 0.01
                            )
    # Train it
    clf.fit(X_train, y_train, features=features, prediction_name="(income:>50K)")

    # Prepare the test set
    df_test = pd.DataFrame(X_test, columns=features)
    df_test['income'] = y_test

    # Compute our model's predictions
    df_test["predictions"] = clf.predict(X_test)
    #print(clf.predict(X_test))
    #print("predict with scores :")
    #print(clf.predict_with_scores(X_test))
    cm = ConfusionMatrix(df_test["gender_Female"], df_test["gender_Male"], df_test["predictions"], df_test["income"])
    cm_minority, cm_majority = cm.get_matrix()
    fm = Metric(cm_minority, cm_majority)

    acc = clf.score(X_test, y_test)
    unf = fm.statistical_parity()
    length = len(clf.rl_.rules)
    print("Fold ", foldIndex, " computed rule list : \n", clf.rl_)
    print("Fold ", foldIndex, " accuracy ", acc)
    print("Fold ", foldIndex, " unfairness ", unf)
    accuracy.append(acc)
    unfairness.append(unf)

print("=========> Accuracy (average)= ", np.average(accuracy))
print("=========> Unfairness (average)= ", np.average(unfairness))