import pandas as pd 
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 statistical_parity, 2 predictive_parity, 3 predictive_equality, 4 equal_opportunity')
parser.add_argument('--dataset', type=str, default="compas", help='either adult or compas')
parser.add_argument('--maxTime', type=int, default=600, help='filtering : 0 no, 1 prefix, 2 all extensions')

args = parser.parse_args()

#epsilon_range = np.arange(0.95, 1.00, 0.001) #0.001
#base = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93,0.935, 0.94,0.945]

#epsilon_range = base + list(epsilon_range)
#epsL = [0.7, 0.8, 0.9, 0.95, 0.975, 0.98, 0.985, 0.99, 0.995]#, 0.999]#[round(x,3) for x in epsilon_range] #60 values

datasets=["adult", "compas"]

epsilons = [0.9, 0.95, 0.98, 0.99, 0.995] #0.7, 0.8, 0.9, 
seeds = []
for i in range(0,20):
    seeds.append(i)

metrics=[1,3,4,5]

max_times=[600] #60,300,1800

filteringModes = [0, 1, 2]

cart_product = [args.dataset, args.metric, args.maxTime]
dataset = cart_product[0]
fairnessMetric = cart_product[1]
max_time = cart_product[2]
policy = "bfs"


optList = {}
for f in filteringModes:
    optList[f] = []


for epsilon in epsilons:
    resList = {}
    for f in filteringModes:
        resList[f] = []

    for seed in seeds: # for each "instance"
        for filteringMode in filteringModes: # for each filtering strategy/policy
            try:
                fileName = './results/%s_eps%f_metric%d_LB%d_%s_tLimit%d_single_seed%d.csv' %(dataset, epsilon, fairnessMetric, filteringMode, policy, max_time, seed)
                fileContent = pd.read_csv(fileName)
                seedVal = fileContent.values[0][0]
                if seedVal != seed:
                    print("Seed number does not match file name, exiting.")
                    exit()
                # Retrieve all information
                '''trainAcc = fileContent.values[0][1]
                trainUnf = fileContent.values[0][2]
                trainObjF = fileContent.values[0][3]
                testAcc = fileContent.values[0][4]
                testUnf = fileContent.values[0][5]
                nodesExplored = fileContent.values[0][6]
                cacheSize = fileContent.values[0][7]
                length = fileContent.values[0][8]
                cpuTime = fileContent.values[0][9]
                solvingStatus = fileContent.values[0][10]
                model = fileContent.values[0][11]'''

                resList[filteringMode].append(fileContent.values[0])

            except FileNotFoundError as not_found:
                print("[Metric %d, Suffix %s] Error: Some result files probably miss." %(fairnessMetric, suffix))
                print("Missing file: ", not_found.filename)
            
    for filteringMode in filteringModes:
        #print("Filtering mode %d" %filteringMode)
        statusList = {}
        for ares in resList[filteringMode]:
            if not ares[10] in statusList.keys():
                statusList[ares[10]] = 1
            else:
                statusList[ares[10]] += 1
        if 'OPT' in statusList.keys():
            nbOpt = statusList['OPT']
        else:
            nbOpt = 0
        nbOpt = nbOpt/len(seeds)
        optList[filteringMode].append(nbOpt)
        #print(statusList)

from matplotlib import pyplot as plt

shapes = ['o', 'x', 'x']
for filteringMode in filteringModes:
    #plt.scatter(epsilons, optList[filteringMode], label="filtering mode %d" %filteringMode, marker=shapes[filteringMode])
    plt.plot(epsilons, optList[filteringMode], label="filtering mode %d" %filteringMode, marker=shapes[filteringMode])
    plt.legend()
plt.title("#Instances solved to optimality as a function of epsilon (metric %d, tLimit= %d s)" %(fairnessMetric, max_time))
plt.show()