import pandas as pd 
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 statistical_parity, 2 predictive_parity, 3 predictive_equality, 4 equal_opportunity')
parser.add_argument('--dataset', type=str, default="compas", help='either adult or compas')
parser.add_argument('--maxTime', type=int, default=1200, help='filtering : 0 no, 1 prefix, 2 all extensions')

args = parser.parse_args()

#epsilon_range = np.arange(0.95, 1.00, 0.001) #0.001
#base = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93,0.935, 0.94,0.945]

#epsilon_range = base + list(epsilon_range)
#epsL = [0.7, 0.8, 0.9, 0.95, 0.975, 0.98, 0.985, 0.99, 0.995]#, 0.999]#[round(x,3) for x in epsilon_range] #60 values

datasets=["adult", "compas"]

epsilons = [0.90, 0.95, 0.99]#[0.9, 0.95, 0.99]#[0.70, 0.75, 0.80, 0.85, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]#[0.9, 0.95, 0.98, 0.99, 0.995] #0.7, 0.8, 0.9, 
seeds = []
n_seeds = 100
for i in range(0,n_seeds):
    if not i in []:
        seeds.append(i)

metrics=[1,3,4,5]


filteringModes = [0, 1, 2]
readFiles = 0
cart_product = [args.dataset, args.metric, args.maxTime]
dataset = cart_product[0]
fairnessMetric = cart_product[1]
max_time = cart_product[2]
policy = "bfs"
if args.dataset == "compas":
    folderPrefix= "results_v3_compas_broadwell/"#"results-4Go/" #"results-2.5Go/"
elif args.dataset == "german_credit":
    folderPrefix= "result_v1_german_broadwell/results_same_arch_4Go_german/"#"results-4Go/" #"results-2.5Go/"
elif args.dataset == "german_credit2":
    folderPrefix= "result_v1_german2_broadwell/results_same_arch_4Go_german_v2/"
archSuffix = "_broadwell"
scoreList = []
optList = {}
scoreList = {}
cacheList = {}
exploredList = {}

for f in filteringModes:
    optList[f] = []
    scoreList[f] = []
    cacheList[f] = []
    exploredList[f] = []


for epsilon in epsilons:
    resList = {}
    for f in filteringModes:
        resList[f] = []

    for seed in seeds: # for each "instance"
        for filteringMode in filteringModes: # for each filtering strategy/policy
            try:
                fileName = './results/%s%s_eps%f_metric%d_LB%d_%s_tLimit%d_single_seed%d%s.csv' %(folderPrefix,dataset, epsilon, fairnessMetric, filteringMode, policy, max_time, seed, archSuffix)
                # try to open the three so that if one misses all are ignored
                fileNameU = './results/%s%s_eps%f_metric%d_LB%d_%s_tLimit%d_single_seed%d%s.csv' %(folderPrefix,dataset, epsilon, fairnessMetric, 0, policy, max_time, seed, archSuffix)
                fileContentU = pd.read_csv(fileNameU)
                fileNameU = './results/%s%s_eps%f_metric%d_LB%d_%s_tLimit%d_single_seed%d%s.csv' %(folderPrefix,dataset, epsilon, fairnessMetric, 1, policy, max_time, seed, archSuffix)
                fileContentU = pd.read_csv(fileNameU)
                fileNameU = './results/%s%s_eps%f_metric%d_LB%d_%s_tLimit%d_single_seed%d%s.csv' %(folderPrefix,dataset, epsilon, fairnessMetric, 2, policy, max_time, seed, archSuffix)
                fileContentU = pd.read_csv(fileNameU)
                fileContent = pd.read_csv(fileName)
                seedVal = fileContent.values[0][0]
                readFiles+=1
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

    for filteringMode in filteringModes:
        # OPT
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
        nbOpt = nbOpt/len(resList[0])
        optList[filteringMode].append(nbOpt)
    # OBJ
    allscores = [[], [], []]
    
    for aresindex in range(len(resList[0])):
        objsList = []
        for f in filteringModes:
            objsList.append(resList[f][aresindex][3])
        #print(objsList)
        ub = max(objsList)
        lb = min(objsList)
        for f in filteringModes:
            if objsList[f] == lb:
                score = 1.0
            else:
                score = 0.0
            #score = (ub - objsList[f]+1) / (ub - lb + 1) # objs[f]
            #score = objsList[f] / lb
            allscores[f].append(score)
            #if score != 1:
            #    print(score)
    for f in filteringModes:
        scoreList[f].append(np.mean(allscores[f]))
    # CACHE
    allcaches = [[], [], []]
    for aresindex in range(len(resList[0])):
        icachesList = []
        objsList = []
        for f in filteringModes:
            icachesList.append(resList[f][aresindex][7])
            objsList.append(resList[f][aresindex][3])
        if not (objsList[0] == objsList[1] == objsList[2]):
            continue
        ub = max(icachesList)
        for f in filteringModes:
                owncache = icachesList[f]/ub # objs[f]
                #if score != 1:
                #    print(score)
                allcaches[f].append(owncache)
    for f in filteringModes:
        cacheList[f].append(np.mean(allcaches[f]))
    # EXPLORED NODES
    allexplored = [[], [], []]
    for aresindex in range(len(resList[0])):
        iexploredList = []
        objsList = []
        for f in filteringModes:
            iexploredList.append(resList[f][aresindex][6])
            objsList.append(resList[f][aresindex][3])
        if not (objsList[0] == objsList[1] == objsList[2]):
            continue
        ub = max(iexploredList)
        for f in filteringModes:
                ownexpl = iexploredList[f]/ub # objs[f]
                #if score != 1:
                #    print(score)
                allexplored[f].append(ownexpl)
    for f in filteringModes:
        exploredList[f].append(np.mean(allexplored[f]))

def proc_number(n):
    return round(n, 2)

fileName = './figures/figures_paper/table_dataset%s_metric%d_policy%s_maxTime%d.csv' %(dataset, fairnessMetric, policy, max_time)
with open(fileName, mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # Old way, including cache and #explored data :
    #csv_writer.writerow(['epsilon', '%OPT', 'Obj', 'Cache', 'Explored', '%OPT', 'Obj', 'Cache', 'Explored', '%OPT', 'Obj', 'Cache', 'Explored'])
    #for i in range(len(epsilons)):
    #    csv_writer.writerow([epsilons[i], proc_number(optList[0][i]),  proc_number(scoreList[0][i]), proc_number(cacheList[0][i]), proc_number(exploredList[0][i]), proc_number(optList[1][i]), proc_number(scoreList[1][i]), proc_number(cacheList[1][i]), proc_number(exploredList[1][i]), proc_number(optList[2][i]), proc_number(scoreList[2][i]), proc_number(cacheList[2][i]), proc_number(exploredList[2][i]) ])
    # New way (only OPT and Obj) :
    csv_writer.writerow(['epsilon', '%OPT', 'Obj', '%OPT', 'Obj', '%OPT', 'Obj'])
    for i in range(len(epsilons)):
        csv_writer.writerow([epsilons[i], proc_number(optList[0][i]),  proc_number(scoreList[0][i]), proc_number(optList[1][i]), proc_number(scoreList[1][i]), proc_number(optList[2][i]), proc_number(scoreList[2][i])])
print("Read ", readFiles, " files")