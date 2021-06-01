#import pandas as pd
import os.path
import argparse


parser = argparse.ArgumentParser(description='Analysis of FairCORELS results')
parser.add_argument('--dataset', type=str, default="compas", help='either adult or compas')

args = parser.parse_args()

dataset=args.dataset
cart_product = []
policy = "bfs"

folderPrefix= "results_learning_quality_%s/results_learning_quality_%s/" %(dataset,dataset)#"results_run_broadwell/"#"results-4Go/" #"results-2.5Go/"
archSuffix = "_broadwell"
# -----------------------------------------------------
datasets= ["compas"]#["adult", "compas"]

# epsilons for learning quality experiments:
# epsilons = [0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.9825, 0.985, 0.9875, 0.99, 0.9925, 0.995, 0.9975] #0.7, 0.8, 0.9,

if dataset == 'compas':
    epsilons = [0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.9825, 0.985, 0.9875, 0.99, 0.9925, 0.995, 0.9975] #0.7, 0.8, 0.9,
    max_times=[1200] 
    print("hello")
elif dataset == "german_credit":
    epsilons = [0.90, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.982, 0.984, 0.986, 0.988, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999] #0.7, 0.8, 0.9,
    max_times=[2400] 

missingSeeds = []
metrics=[1, 3, 4, 5]#, 4, 5]

filteringModes = [0, 1, 2]
# -----------------------------------------------------
print(max_times)
for d in datasets:
    for e in epsilons:
        #for s in seeds:
        for m in metrics:
            for mt in max_times:
                for fm in filteringModes:
                    cart_product.append([d,e,m,mt,fm])
nb_expes=len(cart_product)
expe_ids = []
for i in range(nb_expes):
    expe_ids.append(i)
print("Needed to run %d expes." %nb_expes)
cnt = 0
missingcnt = 0
torerun = []
runsParams = []
for expe_id in expe_ids:
    #dataset = cart_product[expe_id][0]
    epsilon = cart_product[expe_id][1]
    fairnessMetric = cart_product[expe_id][2]
    max_time = cart_product[expe_id][3]
    filteringMode = cart_product[expe_id][4]
    fileName = './results/%s%s_eps%f_metric%d_LB%d_%s_tLimit%d_broadwell.csv' %(folderPrefix, dataset, epsilon, fairnessMetric, filteringMode, policy, max_time)
    if not os.path.isfile(fileName):
        print("Missing file : ", fileName)
        missingcnt+=1
        if not expe_id in torerun:
            torerun.append(expe_id)    
        runsParams.append([dataset, epsilon, fairnessMetric, max_time, filteringMode])        
    else:
        cnt+=1
            
        '''try:
            fileContent = pd.read_csv(fileName)
        except FileNotFoundError as not_found:
            #print("Missing seed %d, max_time=%d, metric= %d, epsilon=%lf, expe_id=%d" %(seed, max_time, fairnessMetric, epsilon, expe_id))
            if not expe_id in torerun:
                torerun.append(expe_id)'''
        

if len(torerun) == 0:
    print("All files found.")
else:
    print("Expes to re-run are: ")
    # print without white space so that copy-paste is easy
    print(str(torerun).replace(" ", ""))
    print("total = %d expes to re-run" %len(torerun))
print("Found %d files" %cnt)
if missingcnt > 0:
    print("Missing %d files" %missingcnt)


print(runsParams)
