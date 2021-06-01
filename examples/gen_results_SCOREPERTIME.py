import pandas as pd 
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 statistical_parity, 2 predictive_parity, 3 predictive_equality, 4 equal_opportunity')
parser.add_argument('--epsilon', type=float, default=0.995, help='epsilon value (min fairness acceptable) for epsilon-constrained method')
parser.add_argument('--dataset', type=str, default="compas", help='either adult or compas')
parser.add_argument('--save', type=bool, default=False, help='save plot png into figures folder')
parser.add_argument('--epsilons', nargs="+", default=[], help='epsilon value (min fairness acceptable) for epsilon-constrained method')
parser.add_argument('--show', type=int, default=1, help='display plot')

args = parser.parse_args()

#epsilon_range = np.arange(0.95, 1.00, 0.001) #0.001
#base = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93,0.935, 0.94,0.945]

#epsilon_range = base + list(epsilon_range)
#epsL = [0.7, 0.8, 0.9, 0.95, 0.975, 0.98, 0.985, 0.99, 0.995]#, 0.999]#[round(x,3) for x in epsilon_range] #60 values

datasets=["adult", "compas"]

epsilons = args.epsilons
if len(epsilons) == 0:
    epsilons.append(args.epsilon)
n_seeds = 100
seeds = []
for i in range(0,n_seeds):
    seeds.append(i)

metrics=[1,3,4,5]

max_times=[120, 300, 400, 500, 600, 900, 1200] #60,300,1800

filteringModes = [0, 1, 2]

cart_product = [args.dataset, args.metric, args.epsilon]
dataset = cart_product[0]
fairnessMetric = cart_product[1]
epsilon = cart_product[2]
policy = "bfs"
folderPrefix= "results-4Go/" #"results-2.5Go/"
archSuffix = "_broadwell"

if dataset == "compas":
    folderPrefix= "results_v3_compas_broadwell/"#"results-4Go/" #"results-2.5Go/"
elif dataset == "german_credit":
    folderPrefix= "result_v1_german_broadwell/results_same_arch_4Go_german/" #"results_run_broadwell/"#"results-4Go/" #"results-2.5Go/"
    max_times=[120, 300, 400, 500, 600, 900, 1200, 1800, 2400] 

readCnt = 0
optList = {}
for f in filteringModes:
    optList[f] = []

# compute scores for each instance
scoresAll={}
for f in filteringModes:
    scoresAll[f]=[]
    for max_time in max_times:
        scoresAll[f].append([]) # for each time period, score for every instance

for epsilonStr in epsilons:
    epsilon = float(epsilonStr)
    for seed in seeds: # for each instance
        resList = {}
        for f in filteringModes:
            resList[f] = [] # for each filtering algo, for an instance, obj function for different time limits

        for max_time in max_times:
            for filteringMode in filteringModes: # for each filtering strategy/policy
                try:
                    fileName = './results/%s%s_eps%f_metric%d_LB%d_%s_tLimit%d_single_seed%d%s.csv' %(folderPrefix, dataset, epsilon, fairnessMetric, filteringMode, policy, max_time, seed, archSuffix)
                    fileContent = pd.read_csv(fileName)
                    readCnt+=1
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

        

        # compute lb and ub
        ub = -1
        lb = 2
        for index, max_time in enumerate(max_times):
            for f in filteringModes:
                currObj = resList[f][index][3]
                if ub < currObj:
                    ub = currObj
                if lb > currObj:
                    lb = currObj

        # compute scores
        #scoresArr = []
        for index, max_time in enumerate(max_times):
            for f in filteringModes:
                score = (ub -resList[f][index][3]+1e-20) / (ub - lb + 1e-20) # objs[f]
                #scoresArr.append(score)
                #if score != 1:
                #    print(score)
                scoresAll[f][index].append(score)
        #print(scoresArr)
    
#print(scoresAll)
# put average scores for all time limits
for f in filteringModes:
    for index, max_time in enumerate(max_times):
        #print(scoresAll[f])
        #print(np.average(scoresAll[f]))
        optList[f].append(np.average(scoresAll[f][index])) # averages for every time period (over scores for all instances)



from matplotlib import pyplot as plt

shapes = ['o', 'x', 'x']
print("Analysed files for epsilon in ", epsilons)
print("Read %d files" %readCnt)

mode = "inverse" #"normal"
for filteringMode in filteringModes:
    #plt.scatter(epsilons, optList[filteringMode], label="filtering mode %d" %filteringMode, marker=shapes[filteringMode])
    #print(optList[filteringMode])
    if filteringMode == 0:
        label = "no filtering"
    elif filteringMode == 1:
        label = "lazy filtering"
    elif filteringMode == 2:
        label = "eager filtering"
    if mode == "normal":
        plt.plot(max_times, optList[filteringMode], label=label, marker=shapes[filteringMode])
    elif mode == "inverse":
        plt.plot(optList[filteringMode], max_times, label=label, marker=shapes[filteringMode])
        plt.xlabel("Objective function quality (score)")
        plt.ylabel("CPU time (s)")
    else:
        print("Unknown plot mode (%s), exiting." %mode)
        exit()
    plt.legend()
#plt.title("Reached objective function as a function of running time (metric %d) - epsilon=%f" %(fairnessMetric, epsilon))


if args.save:
    if len(epsilons)==1:
        plt.savefig('./figures/figures_paper/%s_obj_per_time_metric%d_epsilon%f.pdf' %(dataset, fairnessMetric, epsilons[0]))
    else:
         plt.savefig('./figures/figures_paper/%s_obj_per_time_metric%d_epsilonsList.pdf' %(dataset, fairnessMetric), bbox_inches='tight')

if args.show == 1:
    plt.show()
else:
    plt.close()