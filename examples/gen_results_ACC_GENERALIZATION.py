import pandas as pd 
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

def proc_number(n):
    return round(n, 3)


plotModesList = ['unf_perf_gene', 'unf_gene', 'acc_perf_gene', 'training_acc', 'acc_gene', 'test_acc', 'unf_violation', 'summary_table']
parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 statistical_parity, 2 predictive_parity, 3 predictive_equality, 4 equal_opportunity')
parser.add_argument('--dataset', type=str, default="compas", help='either adult or compas')
parser.add_argument('--save', type=bool, default=False, help='save plot png into figures folder')
parser.add_argument('--show', type=int, default=1, help='display plot')
parser.add_argument('--mode', type=str, default='training_acc', help='Plot mode. Must be one of: %s' %str(plotModesList))

args = parser.parse_args()
fairnessMetric = int(args.metric)
filteringModes = [0, 1, 2]
epsilonList = []
dataset = args.dataset
policy = "bfs"
min_eps = 0
folderPrefix= "results_learning_quality_%s/results_learning_quality_%s/" %(dataset,dataset)#"results_run_broadwell/"#"results-4Go/" #"results-2.5Go/"
filteringModesList = ["no filtering", "lazy filtering", "eager filtering"]
if dataset == "compas":
    #folderPrefix= "results_learning_quality_compas/results_learning_quality_compas/"#"results-4Go/" #"results-2.5Go/"
    epsilons = [0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.9825, 0.985, 0.9875, 0.99, 0.9925, 0.995, 0.9975] #0.7, 0.8, 0.9,
    max_time = 1200
    if args.mode == 'summary_table':
        #epsilons =  0.90, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.9825, 0.985, 0.9875, 0.99, 0.9925, 0.995, 0.9975
        #[0.90, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995]
        min_eps=0.90#min(epsilons) # useful for summary table only
elif dataset == "german_credit":
    #folderPrefix= "result_v1_german_broadwell/results_same_arch_4Go_german/" #"results_run_broadwell/"#"results-4Go/" #"results-2.5Go/"
    epsilons = [0.90, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.982, 0.984, 0.986, 0.988, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999] #0.7, 0.8, 0.9,
    max_time = 2400 
    if args.mode == 'summary_table':
        #epsilons = [0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999] #0.7, 0.8, 0.9,
        min_eps=0.90#min(epsilons) # useful for summary table only

archSuffix = "broadwell"

train_accuracyList = []
test_accuracyList = []
train_unfairnessList = []
test_unfairnessList = []
best_test_acc = []
best_train_acc = []
test_unf_violation = []
for f in filteringModes:
    train_accuracyList.append([])
    test_accuracyList.append([])
    train_unfairnessList.append([])
    test_unfairnessList.append([])
    best_test_acc.append([])
    best_train_acc.append([])
    test_unf_violation.append([])
index = 0
for epsilon in epsilons:
    try: # this block is only to be able to get plots with partial results
        for filteringMode in filteringModes:
            fileName = './results/%s%s_eps%f_metric%d_LB%d_%s_tLimit%d_%s.csv' %(folderPrefix,dataset, epsilon, fairnessMetric, filteringMode, policy, max_time, archSuffix)
            fileContent = pd.read_csv(fileName)
    except:
        print("Ignoring results for epsilon = %f" %epsilon)
        continue
    epsilonList.append(epsilon)
    for filteringMode in filteringModes:
        fileName = './results/%s%s_eps%f_metric%d_LB%d_%s_tLimit%d_%s.csv' %(folderPrefix,dataset, epsilon, fairnessMetric, filteringMode, policy, max_time, archSuffix)
        fileContent = pd.read_csv(fileName)
        trainAccs = []
        testAccs = []
        trainUnfs = []
        testUnfs = []
        for fold in range(5): # iterate over the 5 folds
            foldLineContent = fileContent.values[fold][0].strip('][').split(', ') # parce que je mets une str entière dans la première case du csv :(
            trainAccs.append(float(foldLineContent[1]))
            testAccs.append(float(foldLineContent[4]))
            trainUnfs.append(float(foldLineContent[2]))
            testUnfs.append(float(foldLineContent[5]))
        #print(trainAccs)
        train_accuracyList[filteringMode].append(np.average(trainAccs))
        test_accuracyList[filteringMode].append(np.average(testAccs))
        train_unfairnessList[filteringMode].append(np.average(trainUnfs))
        test_unfairnessList[filteringMode].append(np.average(testUnfs))
    # best test accuracy
    if epsilon > min_eps:
        test_accs_tmp = []
        train_accs_tmp = []
        for f in filteringModes:
            test_accs_tmp.append(test_accuracyList[f][index])
            train_accs_tmp.append(train_accuracyList[f][index])
        best_test_accuracy = max(test_accs_tmp)
        best_train_accuracy = max(train_accs_tmp)
        #print("epsilon=", epsilon, ", test_acs_tmp = ", test_accs_tmp)
        for f in filteringModes:
            if test_accuracyList[f][index] == best_test_accuracy:
                best_test_acc[f].append(1)
            else: 
                best_test_acc[f].append(0)
            if train_accuracyList[f][index] == best_train_accuracy:
                best_train_acc[f].append(1)
            else: 
                best_train_acc[f].append(0)
    # unfairness violation
    if epsilon > min_eps:
        for f in filteringModes:
            test_unf_violation[f].append(test_unfairnessList[f][index]-(1-epsilon))
    index += 1
relativeAccuracyDiffs = []
for f in filteringModes:
    relativeAccuracyDiffs.append([])
    for i in range(len(train_accuracyList[f])):
        relativeAccuracyDiffs[f].append((test_accuracyList[f][i]-train_accuracyList[f][i])/train_accuracyList[f][i])
relativeUnfairnessDiffs = []
for f in filteringModes:
    relativeUnfairnessDiffs.append([])
    for i in range(len(train_unfairnessList[f])):
        relativeUnfairnessDiffs[f].append((test_unfairnessList[f][i]-train_unfairnessList[f][i])/train_unfairnessList[f][i])

test_unfairnessViolation = []
for f in filteringModes:
    test_unfairnessViolation.append([])
    for i in range(len(train_unfairnessList[f])):
        test_unfairnessViolation[f].append(test_unfairnessList[f][i]-(1-epsilons[i]))
#print(epsilonList)

mode = args.mode #'acc_perf_gene' #'training_acc' # 'acc_gene'
shapes = ['o', 'x', 'x']
colors = ['tab:blue', 'tab:orange', 'tab:green']

for f in filteringModes:
    #print(train_accuracyList[f])
    if mode == 'acc_gene':
        plt.plot(epsilonList, relativeAccuracyDiffs[f], label=filteringModesList[f], marker=shapes[filteringMode])
    elif mode == 'unf_violation':
        plt.plot(epsilonList, test_unfairnessViolation[f], label=filteringModesList[f], marker=shapes[filteringMode])
    elif mode == 'unf_gene':
        plt.plot(epsilonList, relativeUnfairnessDiffs[f], label=filteringModesList[f], marker=shapes[filteringMode])
    elif mode == 'training_acc':
        plt.plot(epsilonList, train_accuracyList[f], label=filteringModesList[f], marker=shapes[filteringMode])
    elif mode == 'test_acc':
        plt.plot(epsilonList, test_accuracyList[f], label=filteringModesList[f], marker=shapes[filteringMode])
    elif mode == 'acc_perf_gene':
        plt.plot(epsilonList, train_accuracyList[f], label='%s (train)' %filteringModesList[f], color=colors[f], marker=shapes[filteringMode])
        l, = plt.plot(epsilonList, test_accuracyList[f], '--', label='%s (test)' %filteringModesList[f], color=colors[f], marker=shapes[filteringMode])
        dashes = [2]
        l.set_dashes(dashes)
    elif mode == 'unf_perf_gene':
        plt.plot(epsilonList, train_unfairnessList[f], label=filteringModesList[f], color=colors[f], marker=shapes[filteringMode])
        l, = plt.plot(epsilonList, test_unfairnessList[f], '--', label=filteringModesList[f], color=colors[f], marker=shapes[filteringMode])
        dashes = [2]
        l.set_dashes(dashes)
    elif mode == 'summary_table':
        fileName = './figures/figures_paper/table_learningQ_dataset%s_metric%d_policy%s_maxTime%d.csv' %(dataset, fairnessMetric, policy, max_time)
        '''print("best_test_acc : ")
        print(best_test_acc)
        print("test_unf_violation :")
        print(test_unf_violation[0])
        print(test_unf_violation[1])
        print(test_unf_violation[2])'''

        with open(fileName, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Old way, including cache and #explored data :
            #csv_writer.writerow(['epsilon', '%OPT', 'Obj', 'Cache', 'Explored', '%OPT', 'Obj', 'Cache', 'Explored', '%OPT', 'Obj', 'Cache', 'Explored'])
            #for i in range(len(epsilons)):
            #    csv_writer.writerow([epsilons[i], proc_number(optList[0][i]),  proc_number(scoreList[0][i]), proc_number(cacheList[0][i]), proc_number(exploredList[0][i]), proc_number(optList[1][i]), proc_number(scoreList[1][i]), proc_number(cacheList[1][i]), proc_number(exploredList[1][i]), proc_number(optList[2][i]), proc_number(scoreList[2][i]), proc_number(cacheList[2][i]), proc_number(exploredList[2][i]) ])
            # New way (only OPT and Obj) :
            full = 2
            if full ==1: # with unfairness violation median value
                csv_writer.writerow(['min_epsilon(total_vals)', 'test_acc_percentbest(average)', 'test_unf_violation(average)', 'test_unf_violation(median)', 'test_acc_percentbest(average)', 'test_unf_violation(average)', 'test_unf_violation(median)', 'test_acc_percentbest(average)', 'test_unf_violation(average)', 'test_unf_violation(median)'])
                csv_writer.writerow(['%f (%s)' %(min_eps, len(best_test_acc[0])), proc_number(np.average(best_test_acc[0])), proc_number(np.average(test_unf_violation[0])), proc_number(np.median(test_unf_violation[0])), proc_number(np.average(best_test_acc[1])), proc_number(np.average(test_unf_violation[1])), proc_number(np.median(test_unf_violation[1])), proc_number(np.average(best_test_acc[2])), proc_number(np.average(test_unf_violation[2])), proc_number(np.median(test_unf_violation[2]))])
            elif full ==0:
                csv_writer.writerow(['min_epsilon(total_vals)', 'test_acc_percentbest(average)', 'test_unf_violation(average)', 'test_acc_percentbest(average)', 'test_unf_violation(average)', 'test_acc_percentbest(average)', 'test_unf_violation(average)'])
                csv_writer.writerow(['%f (%s)' %(min_eps, len(best_test_acc[0])), proc_number(np.average(best_test_acc[0])), proc_number(np.average(test_unf_violation[0])), proc_number(np.average(best_test_acc[1])), proc_number(np.average(test_unf_violation[1])), proc_number(np.average(best_test_acc[2])), proc_number(np.average(test_unf_violation[2]))]) 
            elif full ==2: # with training accuracy data
                csv_writer.writerow(['min_epsilon(total_vals)', 'train_acc_percentbest(average)', 'test_acc_percentbest(average)', 'test_unf_violation(average)', 'train_acc_percentbest(average)', 'test_acc_percentbest(average)', 'test_unf_violation(average)', 'train_acc_percentbest(average)', 'test_acc_percentbest(average)', 'test_unf_violation(average)'])
                csv_writer.writerow(['%f (%s)' %(min_eps, len(best_test_acc[0])), proc_number(np.average(best_train_acc[0])), proc_number(np.average(best_test_acc[0])), proc_number(np.average(test_unf_violation[0])), proc_number(np.average(best_train_acc[1])), proc_number(np.average(best_test_acc[1])), proc_number(np.average(test_unf_violation[1])),  proc_number(np.average(best_train_acc[2])),proc_number(np.average(best_test_acc[2])), proc_number(np.average(test_unf_violation[2]))]) 
        exit()
    else:
        print("Unknown plot mode. Valid modes are: ")
        print(plotModesList)
        print("Exiting")
        exit()
plt.legend()
plt.show()
'''
with open('./res-bounds/compile_eps_metric%d.csv' %(fairnessMetric), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['epsilon', 'obj_no_bound', 'obj_bound'])
    for index in range(len(epsilonList)):
        csv_writer.writerow([epsilonList[index], objFListWO[index], objFListW[index]])
'''