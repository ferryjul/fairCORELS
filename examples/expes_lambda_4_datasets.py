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
from mpi4py import MPI

N_ITER = 1*10**8 # The maximum number of nodes in the prefix tree
sensitive_attr_column = 0
unsensitive_attr_column = 1
filteringMode = 0

# parser initialization
parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--expe', type=int, default=0, help='expe id for cartesian product')

args = parser.parse_args()
expe_id = args.expe

cart_product = []

# -----------------------------------------------------
datasets= ["adult_income"]#, "compas", "default_credit", "marketing"]#["adult", "compas"]

epsListBase = np.arange(0.7, 0.85, 0.05)
epsList0 = np.arange(0.86, 0.92, 0.01)
epsList1 =  np.arange(0.925, 0.98, 0.005)
epsList2 =  np.arange(0.981, 1.0, 0.001)
epsList = np.concatenate([epsListBase, epsList0, epsList1, epsList2])
metrics=[1, 3, 4, 5] #, 3, 4, 5]#,3,4,5]

lambdaValues = [1e-3, 5e-3, 1e-2, 5e-2]
max_times=[1200] 

predictions = dict()
predictions["adult_income"] = "(>50K:yes)"
predictions["compas"] = "(redivism_2_years:yes)"
predictions["default_credit"] = "(default_next_month:yes)"
predictions["marketing"] = "(subscribe:yes)"

# -----------------------------------------------------

for d in datasets:
    for e in epsList:
        for m in metrics:
            for mt in max_times:
                for fm in lambdaValues:
                    cart_product.append([d,e,m,mt,fm])

print("cart product has len ", len(cart_product))

dataset = cart_product[expe_id][0]
epsilon = cart_product[expe_id][1]
fairnessMetric = cart_product[expe_id][2]
max_time = cart_product[expe_id][3]
lambdaValue = cart_product[expe_id][4]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Process 0 checks if there are as many workers as expected
if rank == 0:
    if size != 5:
        print("Expected 5 workers for the 5 folds, got: ", size)
        print("Exiting")
        exit()
    else:
        print("Init OK!")
max_memory = 4000
policy="bfs"

X, y, features, prediction = load_from_csv("./data/%s_fullRules.csv" %dataset)#("./data/adult_full.csv") # Load the dataset

# creating k-folds (all workers proceed)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

i=0
for train_index, test_index in kf.split(X):
    if rank == i:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    i +=1

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
                        c=lambdaValue, # sparsity regularization parameter
                        max_card=1, # one rule = one attribute
                        policy=policy, # exploration heuristic
                        bfs_mode=2, # type of BFS: objective-aware
                        mode=3, # epsilon-constrained mode
                        filteringMode=filteringMode,
                        forbidSensAttr=False,
                        map_type="prefix",
                        fairness=fairnessMetric, 
                        epsilon=epsilon, # fairness constrait
                        verbosity=[], # don't print anything
                        maj_vect=unSensVect_train, # vector defining unprotected group
                        min_vect=sensVect_train, # vector defining protected group
                        min_support = 0.05
                        )

start = time.clock()

# Train it
clf.fit(X_train_unprotected, y_train, features=features[2:], prediction_name=predictions[dataset], time_limit = max_time, memory_limit=max_memory)# max_evals=100000) # time_limit=8100, 

time_elapsed = time.clock() - start

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
objF = ((1-accTraining) + (lambdaValue*length)) # best objective function reached
exploredBeforeBest = int(clf.nbExplored)
cacheSizeAtExit = int(clf.nbCache)
#return [foldIndex, accTraining, unfTraining, objF, accTest, unfTest, exploredBeforeBest, cacheSizeAtExit, length, time_elapsed,  clf.get_solving_status()]

res = [[rank, accTraining, unfTraining, objF, accTest, unfTest, exploredBeforeBest, cacheSizeAtExit, length, time_elapsed, clf.get_solving_status(), clf.rl_]]

# Gather the results for the 5 folds on process 0
res = comm.gather(res, root=0)


# Process 0 checks the results
if rank == 0:
    if len(res) != 5:
        print("Problem while gathering the results, len(res) = ", len(res), ", expected 5.")
        print("Res is : ", res)
        print("Exiting")
        exit()

# Process 0 saves the results
if rank == 0:
    if max_time is None: 
        fileName = './results_expes_lambda/%s_eps%f_metric%d_%s_broadwell_lambda%f.csv' %(dataset, epsilon, fairnessMetric, policy, lambdaValue)
    else:
        fileName = './results_expes_lambda/%s_eps%f_metric%d_%s_tLimit%d_broadwell_lambda%f.csv' %(dataset, epsilon, fairnessMetric, policy, max_time, lambdaValue)
    with open(fileName, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Fold', 'Training accuracy', 'Training Unfairness(%d)' %fairnessMetric, 'Training objective function', 'Test accuracy', 'Test unfairness', '#Nodes explored for best solution', 'Cache size for best solution', 'Average length', 'CPU running time (s)', 'Solving Status', 'Model'])
        for i in range(len(res)):
            csv_writer.writerow(res[i])
