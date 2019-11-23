import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from collections import Counter

from faircorels import load_from_csv, CorelsClassifier
from metrics import ConfusionMatrix, Metric


N_ITER = 1*10**5

X, y, features, prediction = load_from_csv("./data/adult_undersampled.csv")



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



def trainFold(X_train, y_train, X_test, y_test):

    clf = CorelsClassifier(n_iter=N_ITER, 
                            c=1e-3, 
                            max_card=2, 
                            policy="bfs",
                            bfs_mode=2,
                            mode=3,
                            useUnfairnessLB=True,
                            forbidSensAttr=False,
                            fairness=1, 
                            epsilon=0.0,
                            maj_pos=34, 
                            min_pos=33,
                            verbosity=["rulelist"]
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

    #print("=========> accuracy {}".format(acc))
    #print("=========> unfairness {}".format(unf))

    return [acc, unf]



output = Parallel(n_jobs=-1)(delayed(trainFold)(X_train=fold[0], y_train=fold[1], X_test=fold[2], y_test=fold[3]) for fold in folds)

accuracy = []
unfairness = []

for res in output:
    accuracy.append(res[0])
    unfairness.append(res[1])


print("=========> median accuracy {}".format(np.median(accuracy)))
print("=========> median unfairness {}".format(np.median(unfairness)))

print("=========> mean accuracy {}".format(np.mean(accuracy)))
print("=========> mean unfairness {}".format(np.mean(unfairness)))