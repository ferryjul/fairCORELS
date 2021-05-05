import pandas as pd 
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 statistical_parity, 2 predictive_parity, 3 predictive_equality, 4 equal_opportunity')

args = parser.parse_args()

epsilon_range = np.arange(0.95, 1.00, 0.001) #0.001
base = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93,0.935, 0.94,0.945]

epsilon_range = base + list(epsilon_range)
epsL = [round(x,3) for x in epsilon_range] #60 values

fairnessMetric = args.metric

epsilonList = []
objFListDelta = []
relCacheSize = []
relExplored = []

for epsilon in epsL:
    if epsilon == 1.0:
        continue
    epsilonList.append(epsilon)
    dataNoBound = pd.read_csv("./results/eps%f_metric%d_LB0-prefix.csv" %(epsilon, fairnessMetric)) 
    dataBound = pd.read_csv("./results/eps%f_metric%d_LB1-prefix.csv" %(epsilon, fairnessMetric)) 
    objBound = dataBound.values[5][3]
    objNoBound = dataNoBound.values[5][3]
    if objBound == objNoBound: # if same sol we compute average cache improvement
        objFListDelta.append(0)
        boundList = []
        noboundList = []
        for i in range(5):
            boundList.append(dataBound.values[i][7])
            noboundList.append(dataNoBound.values[i][7])
        boundAv = np.average(boundList)
        noboundAv = np.average(noboundList)
        relCacheSize.append(float(boundAv)/float(noboundAv))
        boundList = []
        noboundList = []
        for i in range(5):
            boundList.append(dataBound.values[i][6])
            noboundList.append(dataNoBound.values[i][6])
        boundAv = np.average(boundList)
        noboundAv = np.average(noboundList)
        relExplored.append(float(boundAv)/float(noboundAv))
    else: # else we compute the solution's difference
        objFListDelta.append(objBound - objNoBound)
        relCacheSize.append(0) # non sense to compare cache sizes if objs are not the same so we simply put 0
        relExplored.append(0) # non sense to compare #explored nodes if objs are not the same so we simply put 0
    

with open('./results_merged/compile_eps_metric%d_cacheSize.csv' %(fairnessMetric), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['epsilon', 'obj_bound - obj_nobound', 'relative cache size for best sol', 'relative #nodes explored for best sol'])
    for index in range(len(epsilonList)):
        csv_writer.writerow([epsilonList[index], objFListDelta[index], relCacheSize[index], relExplored[index]])