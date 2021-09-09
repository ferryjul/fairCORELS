import numpy as np
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import argparse
from faircorels import load_from_csv, FairCorelsClassifier, ConfusionMatrix, Metric
import csv

N_ITER = 1*10**7 # The maximum number of nodes in the prefix tree
sensitive_attr_column = 0 # Column of the dataset used to define protected group membership (=> sensitive attribute)
unsensitive_attr_column = 1 

X, y, features, prediction = load_from_csv("./data/compas_rules_full_single.csv")#("./data/adult_full.csv") # Load the dataset

print("Sensitive attribute is ", features[sensitive_attr_column])
print("Unsensitive attribute is ", features[unsensitive_attr_column])

# parser initialization
parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--epsilon', type=float, default=0.0, help='epsilon value (min fairness acceptable) for epsilon-constrained method')
parser.add_argument('--filteringMode', type=int, default=1, help='use filtering : 0  no, 1  yes')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 statistical_parity, 2 predictive_parity, 3 predictive_equality, 4 equal_opportunity, 5 Equalized Odds, 6 Conditional use accuracy equality')
parser.add_argument('--newub', type=int, default=1, help='use new ub computation: 1=yes, 0=no')

max_time = 1200
max_memory=4000

args = parser.parse_args()

epsilon = args.epsilon
fairnessMetric = args.metric
newub = args.newub
bools = [False, True]
filteringMode = args.filteringMode
lambdaParam = 1e-3 # The regularization parameter penalizing rule lists length

# We prepare the folds for our 5-folds cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
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
                            policy="bfs", # exploration heuristic: BFS
                            bfs_mode=2, # type of BFS: objective-aware
                            mode=3, # epsilon-constrained mode
                            filteringMode=filteringMode,
                            forbidSensAttr=False,
                            fairness=fairnessMetric, 
                            map_type="none",
                            epsilon=epsilon, # fairness constrait
                            verbosity=[], # don't print anything
                            maj_vect=unSensVect_train, # vector defining unprotected group
                            upper_bound_filtering=newub,
                            min_vect=sensVect_train, # vector defining protected group
                            min_support = 0.01 
                            )
    # Train it
    clf.fit(X_train_unprotected, y_train, features=features[2:], prediction_name="(recidivism:yes)", time_limit = max_time, memory_limit=max_memory)#, time_limit = 10)# max_evals=100000) # time_limit=8100, 

    # Print the fitted model
    print("Fold ", foldIndex, " :", clf.rl_)

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
    return [foldIndex, accTraining, unfTraining, objF, accTest, unfTest, exploredBeforeBest, cacheSizeAtExit, length]

# Run training/evaluation for all folds using multi-threading
ret = Parallel(n_jobs=1)(delayed(oneFold)(foldIndex, X_fold_data) for foldIndex, X_fold_data in enumerate(folds))

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
    resPerFold[aRes[0]] = [aRes[1], aRes[2], aRes[3], aRes[4], aRes[5], aRes[6], aRes[7], aRes[8]]
with open('./results/faircorels_eps%f_metric%d_LB%d.csv' %(epsilon, fairnessMetric, filteringMode), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Fold#', 'Training accuracy', 'Training Unfairness(%d)' %fairnessMetric, 'Training objective function', 'Test accuracy', 'Test unfairness', '#Nodes explored for best solution', 'Cache size for best solution', 'Average length'])#, 'Fairness STD', 'Accuracy STD'])
    for index in range(5):
        csv_writer.writerow([index, resPerFold[index][0], resPerFold[index][1], resPerFold[index][2], resPerFold[index][3], resPerFold[index][4], resPerFold[index][5], resPerFold[index][6], resPerFold[index][7]])#,fairnessSTD_list_test_tot[i][fairnessMetric-1], accuracySTD_list_test_tot[i]])
    csv_writer.writerow(['Average', np.average(accuracyT), np.average(unfairnessT), np.average(objective_functions), np.average(accuracy), np.average(unfairness), '', ''])#,fairnessSTD_list_test_tot[i][fairnessMetric-1], accuracySTD_list_test_tot[i]])
