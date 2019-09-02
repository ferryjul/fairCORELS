import numpy as np
import pandas as pd
from faircorels import *
from metrics import ConfusionMatrix, Metric
from sklearn.model_selection import train_test_split
import cProfile

RANDOM_STATE=42


df = pd.read_csv("data/adult_full.csv")
df_train, df_test = train_test_split(df, test_size=0.33, stratify=df['income'], random_state=RANDOM_STATE)


y_train = df_train['income']
df_train.drop(labels=['income'], axis=1, inplace=True)
features = list(df_train)
X_train = df_train


y_test = df_test['income']
df_test.drop(labels=['income'], axis=1, inplace=True)
X_test = df_test

#for idx, val in enumerate(features):
#   print("feature {} --- val {}".format(val, idx))




clf = CorelsClassifier(n_iter=1500000, 
                        c=0.0001, 
                        max_card=1, 
                        policy="bfs", 
                        bfs_mode=2, 
                        useUnfairnessLB=True, 
                        fairness=1, 
                        min_pos=19, 
                        maj_pos=20, 
                        epsilon=0.95, 
                        mode=3, 
                        verbosity=["rulelist"])


  
cProfile('clf.fit(X_train, y_train, features=features, prediction_name="income")')


predictions=clf.predict(X_test)
        

cm = ConfusionMatrix(X_test["gender:Female"], X_test["gender:Male"], predictions, y_test)

cm_minority, cm_majority = cm.get_matrix()
fm = Metric(cm_minority, cm_majority)


print("Test accuracy {}".format(clf.score(X_test, y_test)))

print("=========> Statiscal parity {}".format(fm.statistical_parity()))
