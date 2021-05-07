import pandas as pd 
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--metric', type=int, default=1, help='fairness metric: 1 statistical_parity, 2 predictive_parity, 3 predictive_equality, 4 equal_opportunity')

args = parser.parse_args()

#epsilon_range = np.arange(0.95, 1.00, 0.001) #0.001
#base = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93,0.935, 0.94,0.945]

#epsilon_range = base + list(epsilon_range)
epsL = [0.7, 0.8, 0.9, 0.95, 0.975, 0.98, 0.985, 0.99, 0.995]#, 0.999]#[round(x,3) for x in epsilon_range] #60 values

fairnessMetric = args.metric

suffixList = ["_bfs_tLimit600_single"]#"_bfs", "_bfs_tLimit30", "_bfs_tLimit60", "_bfs_tLimit120", "_objective_tLimit120", "_objective_tLimit600"]#, "_tLimit300", "_tLimit600"]

for suffix in suffixList:
    try:
        epsilonList = []
        epsilonListAll = []
        objFListDelta = []
        objFListDeltaAll = []
        relCacheSize = []
        relExplored = []
        relCacheSizeAll = []
        relExploredAll = []
        relTimeAll = []
        relTime = []
        statusL = []
        statusLB1 = []
        statusLB2 = []
        # LB1
        for epsilon in epsL:
            if epsilon == 1.0:
                continue
            epsilonList.append(epsilon)
            dataNoBound = pd.read_csv("./results/faircorels_eps%f_metric%d_LB0%s.csv" %(epsilon, fairnessMetric, suffix)) 
            dataBound = pd.read_csv("./results/faircorels_eps%f_metric%d_LB1%s.csv" %(epsilon, fairnessMetric, suffix)) 
            objBound = dataBound.values[5][3]
            objNoBound = dataNoBound.values[5][3]
            '''statusTmpNB = []
            statusTmpLB1 = []
            statusTmpLB2 = []
            for i in range(5):
                statusTmpNB.append(dataNoBound.values[i][10])
                statusTmpLB1.append(dataBound.values[i][10])
                statusTmpLB2.append(dataBound.values[i][10])
            tempS = statusTmpLB2[0]
            for t in statusTmpLB2:
                if t != tempS:
                    print("need manual analysis, status mismatch. Eps=", epsilon, ", metric=", fairnessMetric, " status are:", statusTmpLB2)
                    exit()
            statusLB2.append(tempS)
            tempS = statusTmpNB[0]
            for t in statusTmpNB:
                if t != tempS:
                    print("need manual analysis, status mismatch. Eps=", epsilon, ", metric=", fairnessMetric, " status are:", statusTmpLB2)
                    exit()
            statusL.append(tempS)
            tempS = statusTmpLB1[0]
            for t in statusTmpLB1:
                if t != tempS:
                    print("need manual analysis, status mismatch. Eps=", epsilon, ", metric=", fairnessMetric, " status are:", statusTmpLB2)
                    exit()
            statusLB1.append(tempS)'''
            if objBound == objNoBound: # if same sol we compute average cache improvement
                objFListDelta.append(0)
                boundList = []
                noboundList = []
                timeBoundList = []
                timeNoBoundList = []
                for i in range(5):
                    boundList.append(dataBound.values[i][7])
                    noboundList.append(dataNoBound.values[i][7])
                    timeBoundList.append(dataBound.values[i][9])
                    timeNoBoundList.append(dataNoBound.values[i][9])
                boundAv = np.average(boundList)
                noboundAv = np.average(noboundList)
                timeBoundAv = np.average(timeBoundList)
                timeNoBoundAv = np.average(timeNoBoundList)
                relTime.append(float(timeBoundAv)/float(timeNoBoundAv))
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
                relTime.append(0)

        # LB2
        for epsilon in epsL:
            if epsilon == 1.0:
                continue
            epsilonListAll.append(epsilon)
            dataNoBound = pd.read_csv("./results/faircorels_eps%f_metric%d_LB0%s.csv" %(epsilon, fairnessMetric, suffix)) 
            dataBound = pd.read_csv("./results/faircorels_eps%f_metric%d_LB2%s.csv" %(epsilon, fairnessMetric, suffix)) 
            objBound = dataBound.values[5][3]
            objNoBound = dataNoBound.values[5][3]
            
            if objBound == objNoBound: # if same sol we compute average cache improvement
                objFListDeltaAll.append(0)
                boundList = []
                noboundList = []
                timeBoundList = []
                timeNoBoundList = []
                
                for i in range(5):
                    boundList.append(dataBound.values[i][7])
                    noboundList.append(dataNoBound.values[i][7])
                boundAv = np.average(boundList)
                noboundAv = np.average(noboundList)
                relCacheSizeAll.append(float(boundAv)/float(noboundAv))
                boundList = []
                noboundList = []
                for i in range(5):
                    boundList.append(dataBound.values[i][6])
                    noboundList.append(dataNoBound.values[i][6])
                    timeBoundList.append(dataBound.values[i][9])
                    timeNoBoundList.append(dataNoBound.values[i][9])
                    
                boundAv = np.average(boundList)
                noboundAv = np.average(noboundList)
                timeBoundAv = np.average(timeBoundList)
                timeNoBoundAv = np.average(timeNoBoundList)
                relTimeAll.append(float(timeBoundAv)/float(timeNoBoundAv))
                boundAv = np.average(boundList)
                noboundAv = np.average(noboundList)
                relExploredAll.append(float(boundAv)/float(noboundAv))
            else: # else we compute the solution's difference
                objFListDeltaAll.append(objBound - objNoBound)
                relCacheSizeAll.append(0) # non sense to compare cache sizes if objs are not the same so we simply put 0
                relExploredAll.append(0) # non sense to compare #explored nodes if objs are not the same so we simply put 0
                relTimeAll.append(0)

        if not(len(epsilonListAll) == len(epsilonList)):
            print("[Metric %d, Suffix %s] Error: #epsilons mismatch between LB1 and LB2. Some result files probably miss." %(fairnessMetric, suffix))
            exit(1)

        with open('./results_merged/compile_eps_metric%d_cacheSize%s.csv' %(fairnessMetric, suffix), mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['', 'LB1', 'LB1', 'LB1', 'LB2', 'LB2', 'LB2'])
            csv_writer.writerow(['epsilon', 'obj_bound - obj_nobound', 'relative cache size for best sol', 'relative #nodes explored for best sol', 'relative time', 'obj_bound - obj_nobound', 'relative cache size for best sol', 'relative #nodes explored for best sol', 'relative time'])
            for index in range(len(epsilonList)):
                csv_writer.writerow([epsilonList[index], objFListDelta[index], relCacheSize[index], relExplored[index], relTime[index], objFListDeltaAll[index], relCacheSizeAll[index], relExploredAll[index], relTimeAll[index]])
            print("[Metric %d, Suffix %s] Success - Generated file :" %(fairnessMetric, suffix), './results_merged/compile_eps_metric%d_cacheSize%s.csv' %(fairnessMetric, suffix))
        '''with open('./results_merged/compile_eps_metric%d_cacheSize%s_forLatex.csv' %(fairnessMetric, suffix), mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #csv_writer.writerow(['', 'LB1', 'LB1', 'LB1', 'LB2', 'LB2', 'LB2'])
            csv_writer.writerow(['epsilon', 'no bound status', 'LB1 status', 'obj_bound - obj_nobound', 'relative cache size for best sol', 'relative #nodes explored for best sol', 'LB2 Status', 'obj_bound - obj_nobound', 'relative cache size for best sol', 'relative #nodes explored for best sol'])
            for index in range(len(epsilonList)):
                csv_writer.writerow([epsilonList[index], statusL[index], objFListDelta[index], relCacheSize[index], relExplored[index], statusLB1[index], objFListDeltaAll[index], relCacheSizeAll[index], relExploredAll[index], statusLB2[index]])
            print("[Metric %d, Suffix %s] Success - Generated file :" %(fairnessMetric, suffix), './results_merged/compile_eps_metric%d_cacheSize%s.csv' %(fairnessMetric, suffix))'''
    except FileNotFoundError as not_found:
        print("[Metric %d, Suffix %s] Error: Some result files probably miss." %(fairnessMetric, suffix))
        print("Missing file: ", not_found.filename)