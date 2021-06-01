import pandas as pd 
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 statistical_parity, 2 predictive_parity, 3 predictive_equality, 4 equal_opportunity')
parser.add_argument('--dataset', type=str, default="compas", help='either adult or compas')
parser.add_argument('--maxTime', type=int, default=600, help='max running time')
parser.add_argument('--displayNbInstances', type=int, default=1, help='display number of instances for which same solution was found')
parser.add_argument('--save', type=bool, default=False, help='save plot png into figures folder')
parser.add_argument('--show', type=int, default=1, help='display plot')
parser.add_argument('--reverseEps', type=int, default=0, help='display plot')

args = parser.parse_args()

#epsilon_range = np.arange(0.95, 1.00, 0.001) #0.001
#base = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93,0.935, 0.94,0.945]

#epsilon_range = base + list(epsilon_range)
#epsL = [0.7, 0.8, 0.9, 0.95, 0.975, 0.98, 0.985, 0.99, 0.995]#, 0.999]#[round(x,3) for x in epsilon_range] #60 values

datasets=["adult", "compas"]

epsilons = [0.70, 0.75, 0.80, 0.85, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995] # [0.9, 0.95, 0.98, 0.99, 0.995] #0.7, 0.8, 0.9, 
seeds = []
n_seeds = 100
for i in range(0,n_seeds):
    seeds.append(i)

metrics=[1,3,4,5]

max_times=[600] #60,300,1800

filteringModes = [0, 1, 2]
proportions = []
cart_product = [args.dataset, args.metric, args.maxTime]
dataset = cart_product[0]
fairnessMetric = cart_product[1]
max_time = cart_product[2]
policy = "bfs"
folderPrefix= "results_v3_compas_broadwell/"#"results-4Go/" #"results-2.5Go/"
archSuffix = "_broadwell"
if dataset == "compas":
    folderPrefix= "results_v3_compas_broadwell/"#"results-4Go/" #"results-2.5Go/"
elif dataset == "german_credit":
    folderPrefix= "result_v1_german_broadwell/results_same_arch_4Go_german/" #"results_run_broadwell/"#"results-4Go/" #"results-2.5Go/"
    epsilons = [0.90, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]
optList = {}
for f in filteringModes:
    optList[f] = []


for epsilon in epsilons:
    resList = {}
    for f in filteringModes:
        resList[f] = []
    proportion_eps = 0
    for seed in seeds: # for each "instance"
        for filteringMode in filteringModes: # for each filtering strategy/policy
            try:
                fileName = './results/%s%s_eps%f_metric%d_LB%d_%s_tLimit%d_single_seed%d%s.csv' %(folderPrefix, dataset, epsilon, fairnessMetric, filteringMode, policy, max_time, seed, archSuffix)
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
                print("Error: Some result files probably miss.")
                print("Missing file: ", not_found.filename)
            
    # compute scores for each instance
    scoresAll={}
    for f in filteringModes:
        scoresAll[f]=[]
    for seedV in seeds:
        # retrieve objective function for each filtering algo
        sizes = []
        objs = []
        for f in filteringModes:
            sizes.append(resList[f][seedV][7])
            objs.append(resList[f][seedV][3])
        #print(sizes)
        best = min(objs)
        M = max(sizes)
        ok = True
        for f in filteringModes:
            if objs[f] != best:
                ok = False
        if ok:
            proportion_eps += 1
            for f in filteringModes: 
                score = (sizes[f]) / max((M),1)
                if f == 0 and score != 1.0:
                    print("score %d = " %f, score)
                scoresAll[f].append(score)
    
    # put average scores for this time limit
    for f in filteringModes:
        optList[f].append(np.average(scoresAll[f]))
    proportions.append(str(len(scoresAll[0])))

from matplotlib import pyplot as plt

if args.reverseEps == 0:
    for i in range(len(epsilons)):
        epsilons[i] = 1.0 - epsilons[i]

shapes = ['o', 'x', 'x']
for filteringMode in filteringModes:
    #plt.scatter(epsilons, optList[filteringMode], label="filtering mode %d" %filteringMode, marker=shapes[filteringMode])
    if filteringMode == 0:
        label = "no filtering"
    elif filteringMode == 1:
        label = "lazy filtering"
    elif filteringMode == 2:
        label = "eager filtering"
    plt.plot(epsilons, optList[filteringMode], label=label, marker=shapes[filteringMode])
    plt.legend()
    
    plt.ylabel("Cache size (normalized score)")
    plt.xlabel("$ε$")
    if args.displayNbInstances == 1:
        for i, txt in enumerate(proportions):
            plt.annotate(proportions[i], (epsilons[i], 1.0))

#plt.title("Cache size for best solution as a function of epsilon (metric %d, tLimit= %d s)" %(fairnessMetric, max_time))
print(proportions)
if args.save:
    if args.reverseEps == 0:
        plt.savefig('./figures/figures_paper/%s_cache_per_eps_metric%d_time%d.pdf' %(dataset, fairnessMetric, max_time), bbox_inches='tight')
    else:
        plt.xlabel("$1-ε$")
        plt.savefig('./figures/figures_paper/%s_cache_per_eps_metric%d_time%d_reverse.pdf' %(dataset, fairnessMetric, max_time), bbox_inches='tight')
if args.show == 1:
    plt.show()