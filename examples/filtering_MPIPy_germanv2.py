import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

# parser initialization
parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
'''parser.add_argument('--epsilon', type=int, default=0, help='epsilon value (min fairness acceptable) for epsilon-constrained method')
parser.add_argument('--filteringMode', type=int, default=0, help='filtering : 0 no, 1 prefix, 2 all extensions')
parser.add_argument('--maxTime', type=int, default=-1, help='filtering : 0 no, 1 prefix, 2 all extensions')
parser.add_argument('--policy', type=str, default="bfs", help='search heuristic - function used to order the priority queue')
parser.add_argument('--maxMemory', type=int, default=-1, help='filtering : 0 no, 1 prefix, 2 all extensions')
parser.add_argument('--dataset', type=str, default="compas", help='either adult or compas')
parser.add_argument('--seed', type=str, default="compas", help='either adult or compas')'''
parser.add_argument('--expe', type=int, default=0, help='expe id for cartesian product')
parser.add_argument('--maxTime', type=int, default=600, help='filtering : 0 no, 1 prefix, 2 all extensions')

args = parser.parse_args()
expe_id = args.expe

cart_product = []

# -----------------------------------------------------
datasets= ["german_credit2"]#["adult", "compas"]

epsilons = [0.90, 0.95, 0.98, 0.99, 0.995, 0.998] #0.7, 0.8, 0.9,
seeds = []
for i in range(0,20):
    seeds.append(i)

metrics=[1, 3, 4, 5] #, 3, 4, 5]#,3,4,5]

#max_times=[120, 300, 600, 1200] #

filteringModes = [0, 1, 2]
max_times=[1800] #120, 300, 400, 500,  

# -----------------------------------------------------

for d in datasets:
    for e in epsilons:
        #for s in seeds:
        for m in metrics:
            for mt in max_times:
                for fm in filteringModes:
                    cart_product.append([d,e,m,mt,fm])

#print("cart product has len ", len(cart_product))

dataset = cart_product[expe_id][0]
epsilon = cart_product[expe_id][1]
fairnessMetric = cart_product[expe_id][2]
max_time = cart_product[expe_id][3]
filteringMode = cart_product[expe_id][4]

def split(container, count):
    return [container[_i::count] for _i in range(count)]


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

'''if COMM.rank == 0:
    jobs = split(seeds, COMM.size)
else:
    jobs = None

jobs = COMM.scatter(jobs, root=0)'''
seed = rank
#print("Expe %d: dataset=%s, epsilon=%f, seed=%d, fairnessMetric=%d, max_time=%d, filteringMode=%d" %(expe_id, dataset, epsilon, seed, fairnessMetric, max_time, filteringMode))
lambdaParam = 1e-3
max_memory = 5000
policy="bfs"

X, y, features, prediction = load_from_csv("./data/%s_rules_full_single.csv" %dataset)#("./data/adult_full.csv") # Load the dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=seed)

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
                        c=lambdaParam, # sparsity regularization parameter
                        max_card=1, # one rule = one attribute
                        policy=policy, # exploration heuristic
                        bfs_mode=2, # type of BFS: objective-aware
                        mode=3, # epsilon-constrained mode
                        filteringMode=filteringMode,
                        forbidSensAttr=False,
                        map_type="none",
                        fairness=fairnessMetric, 
                        epsilon=epsilon, # fairness constrait
                        verbosity=[], # don't print anything
                        maj_vect=unSensVect_train, # vector defining unprotected group
                        min_vect=sensVect_train, # vector defining protected group
                        min_support = 0.05
                        )

start = time.clock()

# Train it
clf.fit(X_train_unprotected, y_train, features=features[2:], prediction_name="(recidivism:yes)", time_limit = max_time, memory_limit=max_memory)# max_evals=100000) # time_limit=8100, 

time_elapsed = time.clock() - start

# Print the fitted model
#print("Fold ", foldIndex, " :", clf.rl_, "(RT: ", time_elapsed, " s)")

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
#return [foldIndex, accTraining, unfTraining, objF, accTest, unfTest, exploredBeforeBest, cacheSizeAtExit, length, time_elapsed,  clf.get_solving_status()]


if max_time is None: 
    fileName = './results_same_arch_4Go_german_v2/%s_eps%f_metric%d_LB%d_%s_single_seed%d_broadwell.csv' %(dataset, epsilon, fairnessMetric, filteringMode, policy, seed)
else:
    fileName = './results_same_arch_4Go_german_v2/%s_eps%f_metric%d_LB%d_%s_tLimit%d_single_seed%d_broadwell.csv' %(dataset, epsilon, fairnessMetric, filteringMode, policy, max_time, seed)
with open(fileName, mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Seed', 'Training accuracy', 'Training Unfairness(%d)' %fairnessMetric, 'Training objective function', 'Test accuracy', 'Test unfairness', '#Nodes explored for best solution', 'Cache size for best solution', 'Average length', 'CPU running time (s)', 'Solving Status', 'Model'])#, 'Fairness STD', 'Accuracy STD'])
    csv_writer.writerow([seed, accTraining, unfTraining, objF, accTest, unfTest, exploredBeforeBest, cacheSizeAtExit, length, time_elapsed, clf.get_solving_status(), clf.rl_])#,fairnessSTD_list_test_tot[i][fairnessMetric-1], accuracySTD_list_test_tot[i]])
