import pandas as pd 
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 statistical_parity, 2 predictive_parity, 3 predictive_equality, 4 equal_opportunity')
parser.add_argument('--epsilon', type=float, default=0.995, help='epsilon value (min fairness acceptable) for epsilon-constrained method')
parser.add_argument('--dataset', type=str, default="compas", help='either adult or compas')

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

max_times=[120, 300, 400, 500, 600] #60,300,1800

filteringModes = [0, 1, 2]

cart_product = [args.dataset, args.metric, args.epsilon]
dataset = cart_product[0]
fairnessMetric = cart_product[1]
epsilon = cart_product[2]
policy = "bfs"


optList = {}
for f in filteringModes:
    optList[f] = []


for max_time in max_times:
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
                print("Error: Some result files probably miss.")
                print("Missing file: ", not_found.filename)

    # compute scores for each instance
    scoresAll={}
    for f in filteringModes:
        scoresAll[f]=[]
    for seedV in seeds:
        # retrieve objective function for each filtering algo
        objs = []
        for f in filteringModes:
            objs.append(resList[f][seedV][3])
        #print("objective functions for instance %d are:" %seedV, objs)
        ub = max(objs)
        lb = min(objs)
        if ub != lb:
            print("inst %d, time %d" %(seedV, max_time))
            print(objs)
        for f in filteringModes:
            score = objs[f] #(objs[f] - lb +1) / (ub - lb + 1)
            scoresAll[f].append(score)
    
    # put average scores for this time limit
    for f in filteringModes:
        #print(scoresAll[f])
        #print(np.average(scoresAll[f]))
        optList[f].append(np.average(scoresAll[f]))



from matplotlib import pyplot as plt

shapes = ['o', 'x', 'x']

mode = "normal" #"normal"
for filteringMode in filteringModes:
    #plt.scatter(epsilons, optList[filteringMode], label="filtering mode %d" %filteringMode, marker=shapes[filteringMode])
    #print(optList[filteringMode])
    if mode == "normal":
        plt.plot(max_times, optList[filteringMode], label="filtering mode %d" %filteringMode, marker=shapes[filteringMode])
    elif mode == "inverse":
        plt.plot(optList[filteringMode], max_times, label="filtering mode %d" %filteringMode, marker=shapes[filteringMode])
    else:
        print("Unknown plot mode (%s), exiting." %mode)
        exit()
    plt.legend()
plt.title("Reached objective function as a function of running time (metric %d) - epsilon=%f" %(fairnessMetric, epsilon))
plt.show()