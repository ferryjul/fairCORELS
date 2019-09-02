import numpy as np
import pandas as pd
from faircorels import *
from metrics import ConfusionMatrix, Metric
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

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


def epsilon_corels(eps=0.95, metric=1, n=1000000):
    print("=========================> epsilon: {}, metric: {}".format(eps, metric))
    #initialize the learning algorithm
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

    # fit 
    clf.fit(X_train, y_train, features=features, prediction_name="income")

    # predictions and accuracy on the training/test set
    ## training set
    predictions_train = clf.predict(X_train)
    acc_train = clf.score(X_train, y_train)

    ## test set
    predictions_test = clf.predict(X_test)
    acc_test = clf.score(X_test, y_test)

    # fairness and accuracy on the training/test set
    ## training set 
    cm_train = ConfusionMatrix(X_train["gender:Female"], X_train["gender:Male"], predictions_train, y_train)
    cm_minority_train, cm_majority_train = cm_train.get_matrix()
    fm_train = Metric(cm_minority_train, cm_majority_train)

    ## test set
    cm_test = ConfusionMatrix(X_test["gender:Female"], X_test["gender:Male"], predictions_test, y_test)
    cm_minority_test, cm_majority_test = cm_test.get_matrix()
    fm_test = Metric(cm_minority_test, cm_majority_test)

    unfairness_train = 0.0
    unfairness_test = 0.0

    if metric == 1:
        unfairness_train = fm_train.statistical_parity()
        unfairness_test = fm_test.statistical_parity()
    if metric == 2:
        unfairness_train = fm_train.predictive_parity()
        unfairness_test = fm_test.predictive_parity()
    if metric == 3:
        unfairness_train = fm_train.predictive_equality()
        unfairness_test = fm_test.predictive_equality()
    if metric == 4:
        unfairness_train = fm_train.equal_opportunity()
        unfairness_test = fm_test.equal_opportunity()

    return [eps, metric, acc_train, acc_test, unfairness_train, unfairness_test]


def beta_corels(beta=0.1, metric=1, n=1000000):
    print("=========================> beta: {}, metric: {}".format(beta, metric))

    #initialize the learning algorithm
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

    # fit 
    clf.fit(X_train, y_train, features=features, prediction_name="income")

    # predictions and accuracy on the training/test set
    ## training set
    predictions_train = clf.predict(X_train)
    acc_train = clf.score(X_train, y_train)

    ## test set
    predictions_test = clf.predict(X_test)
    acc_test = clf.score(X_test, y_test)

    # fairness and accuracy on the training/test set
    ## training set 
    cm_train = ConfusionMatrix(X_train["gender:Female"], X_train["gender:Male"], predictions_train, y_train)
    cm_minority_train, cm_majority_train = cm_train.get_matrix()
    fm_train = Metric(cm_minority_train, cm_majority_train)

    ## test set
    cm_test = ConfusionMatrix(X_test["gender:Female"], X_test["gender:Male"], predictions_test, y_test)
    cm_minority_test, cm_majority_test = cm_test.get_matrix()
    fm_test = Metric(cm_minority_test, cm_majority_test)

    unfairness_train = 0.0
    unfairness_test = 0.0

    if metric == 1:
        unfairness_train = fm_train.statistical_parity()
        unfairness_test = fm_test.statistical_parity()
    if metric == 2:
        unfairness_train = fm_train.predictive_parity()
        unfairness_test = fm_test.predictive_parity()
    if metric == 3:
        unfairness_train = fm_train.predictive_equality()
        unfairness_test = fm_test.predictive_equality()
    if metric == 4:
        unfairness_train = fm_train.equal_opportunity()
        unfairness_test = fm_test.equal_opportunity()

    

    return [beta, metric, acc_train, acc_test, unfairness_train, unfairness_test]



#print(epsilon_corels(n=100000, eps=0.97))

#print(beta_corels(n=100000, beta=0.5))

#epsilons = [0.95, 0.96, 0.97, 0.98, 0.99]
#betas = [0.1, 0.2, 0.3, 0.4, 0.5]
#epsilons = np.arange(0.95, 1.001, 0.001)

epsilons = np.linspace(0.95, 1.0, 50)
betas = np.linspace(0.1,0.5, 50)


iterations = 100000

data_eps = {
    'parameter': [],
    'metric': [],
    'acc_train': [],
    'acc_test': [],
    'unfairness_train': [],
    'unfairness_test': [],
}

data_betas = {
    'parameter': [],
    'metric': [],
    'acc_train': [],
    'acc_test': [],
    'unfairness_train': [],
    'unfairness_test': [],
}






for m in [1, 2, 3, 4]:
    eval_metrics = Parallel(n_jobs=-1)(delayed(beta_corels)(beta=beta, metric=m, n=iterations) for beta in betas)
    for res in eval_metrics:
        data_betas['parameter'].append(res[0])
        data_betas['metric'].append(res[1])
        data_betas['acc_train'].append(res[2])
        data_betas['acc_test'].append(res[3])
        data_betas['unfairness_train'].append(res[4])
        data_betas['unfairness_test'].append(res[5])

df_betas = pd.DataFrame(data_betas)

df_betas.to_csv("./output/df_betas.csv", encoding='utf-8', index=False)



for m in [1, 2, 3, 4]:
    eval_metrics = Parallel(n_jobs=-1)(delayed(epsilon_corels)(eps=eps, metric=m, n=iterations) for eps in epsilons)
    for res in eval_metrics:
        data_eps['parameter'].append(res[0])
        data_eps['metric'].append(res[1])
        data_eps['acc_train'].append(res[2])
        data_eps['acc_test'].append(res[3])
        data_eps['unfairness_train'].append(res[4])
        data_eps['unfairness_test'].append(res[5])

df_esp = pd.DataFrame(data_eps)
df_esp.to_csv("./output/df_eps.csv", encoding='utf-8', index=False)