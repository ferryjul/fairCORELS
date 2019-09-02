import numpy as np
import pandas as pd
from faircorels import *
from sklearn.model_selection import KFold
import csv
import os.path

#from sklearn.externals.joblib import Parallel, delayed



df = pd.read_csv("data/adult_full.csv")

y = df['income']
df.drop(labels=['income'], axis=1, inplace=True)
features = list(df)
X = df

#for idx, val in enumerate(features):
#   print("feature {} --- val {}".format(val, idx))

epsilons = np.arange(0.95, 1.001, 0.001)


filename='./output/fairness_1.csv'
file_exists = os.path.isfile(filename)

for eps in epsilons:
    kf = KFold(n_splits=2)
    err = []
    unf = []

    clf = CorelsClassifier(n_iter=2000000, 
                            c=1e-2, 
                            max_card=2,
                            min_support=0.15, 
                            policy="bfs",
                            bfs_mode=2, 
                            fairness=1,
                            rule_min=7,
                            rule_maj=8,
                            min_pos=7,
                            maj_pos=9,
                            epsilon=eps, 
                            mode=1,
                            beta=0.4,
                            verbosity=["rule"])

    for train, test in kf.split(df):
        X_train = np.array(X)[train]
        y_train = np.array(y)[train]
        X_test = np.array(X)[test]
        y_test = np.array(y)[test]
        clf.fit(X_train, y_train, features=features, prediction_name="income")
        accuracy, unfairness = clf.score(X_test, y_test)
        print("accuracy {} --- unfairness {}".format(accuracy, unfairness))
        err.append(1-accuracy)
        unf.append(unfairness)

    err_mean = np.mean(err)
    unf_mean = np.mean(unf)
    print("error {}".format(err_mean))
    print("unfairness {}".format(unf_mean))

    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['parameter', 'error', 'unfairness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow({'parameter': eps, 'error': err_mean, 'unfairness': unf_mean})