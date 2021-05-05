import numpy as np
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import argparse
from faircorels import load_from_csv, FairCorelsBagging, ConfusionMatrix, Metric
import csv

N_ITER = 5*10**5 # The maximum number of nodes in the prefix tree
sensitive_attr_column = 0 # Column of the dataset used to define protected group membership (=> sensitive attribute)
unsensitive_attr_column = 1 

X, y, features, prediction = load_from_csv("./data/compas_rules_full.csv")#("./data/adult_full.csv") # Load the dataset

print("Sensitive attribute is ", features[sensitive_attr_column])
print("Unsensitive attribute is ", features[unsensitive_attr_column])

# parser initialization
parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--epsilon', type=float, default=0.0, help='epsilon value (min fairness acceptable) for epsilon-constrained method')
parser.add_argument('--uselb', type=int, default=0, help='use filtering : 0  no, 1  yes')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 statistical_parity, 2 predictive_parity, 3 predictive_equality, 4 equal_opportunity, 5 Equalized Odds, 6 Conditional use accuracy equality')

args = parser.parse_args()

epsilon = args.epsilon
fairnessMetric = args.metric
bools = [False, True]
useLB = bools[args.uselb]
lambdaParam = 1e-3 # The regularization parameter penalizing rule lists length
n_learners = 8 # Number of base learners
n_workers = 8 # Maximum nhumber of parallel threads for training
baggingRatio = 1.0 # Size of the bootstrap sampling subsamples (ratio to the whole training set)

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

    # Create the FairCorelsBagging object
    clf = FairCorelsBagging(X_train_unprotected, # training set must be given here
                        y_train, # training set must be given here
                        n_learners=n_learners,
                        sample_size=int(X_train_unprotected.shape[0]*baggingRatio), 
                        features=features[2:], 
                        prediction_name="(recidivism:yes)",
                        n_iter=N_ITER, 
                        c=lambdaParam, 
                        max_card=1, 
                        policy="bfs",
                        bfs_mode=2,
                        mode=3,
                        filteringMode=useLB,
                        forbidSensAttr=False,
                        fairness=fairnessMetric, 
                        epsilon=epsilon,
                        verbosity=[], # "rulelist"
                        maj_vect=unSensVect_train, 
                        min_vect=sensVect_train, 
                        min_support = 0.01,
                        baggingVerbose = 1)
    # Train it
    clf.fit(n_workers=n_workers)#, time_limit=5)# max_evals=100000) # time_limit=8100, 

    # Evaluate our model's accuracy
    accTraining = clf.score(X_train_unprotected, y_train)
    accTest = clf.score(X_test_unprotected, y_test)

    # Evaluate our model's fairness
    train_preds = clf.predict(X_train_unprotected)
    unfTraining = compute_unfairness(sensVect_train, unSensVect_train, y_train, train_preds)

    test_preds = clf.predict(X_test_unprotected)
    unfTest = compute_unfairness(sensVect_test, unSensVect_test, y_test, test_preds)

    # Also compute/collect average length
    length = np.average([len(clf.ruleLists[ri].rules)-1 for ri in range(n_learners)])
    
    return [foldIndex, accTraining, unfTraining, accTest, unfTest, length]


# Run training/evaluation for all folds using multi-threading
# Multithreading is included in the FairCorelsBagging object (given parameter n_workers)
# So be aware of it before multithreading this loop
ret = Parallel(n_jobs=1)(delayed(oneFold)(foldIndex, X_fold_data) for foldIndex, X_fold_data in enumerate(folds))

# Unwrap the results
accuracy = [ret[i][3] for i in range(0,5)]
unfairness = [ret[i][4] for i in range(0,5)]
accuracyT = [ret[i][1] for i in range(0,5)]
unfairnessT = [ret[i][2] for i in range(0,5)]
print("=========> Training Accuracy (average)= ", np.average(accuracyT))
print("=========> Training Unfairness (average)= ", np.average(unfairnessT))
print("=========> Test Accuracy (average)= ", np.average(accuracy))
print("=========> Test Unfairness (average)= ", np.average(unfairness))

#Save results in a csv file
resPerFold = dict()
for aRes in ret:
    resPerFold[aRes[0]] = [aRes[1], aRes[2], aRes[3], aRes[4], aRes[5]]
with open('./results/faircorelsbagging_eps%f_metric%d_LB%d_bRatio%f_nmodels%d.csv' %(epsilon, fairnessMetric, useLB, baggingRatio,n_learners), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Fold#', 'Training accuracy', 'Training Unfairness(%d)' %fairnessMetric, 'Test accuracy', 'Test unfairness', 'Average length'])
    for index in range(5):
        csv_writer.writerow([index, resPerFold[index][0], resPerFold[index][1], resPerFold[index][2], resPerFold[index][3], resPerFold[index][4]])
    csv_writer.writerow(['Average', np.average(accuracyT), np.average(unfairnessT), np.average(accuracy), np.average(unfairness)])