#import pandas as pd
import os.path

max_times=[120, 300, 400, 500, 600, 900, 1200] 

cart_product = []
policy = "bfs"
folderPrefix= "results_v3_compas_broadwell/" #"results_run_broadwell/"#"results-4Go/" #"results-2.5Go/"
archSuffix = "_broadwell"
# -----------------------------------------------------
datasets= ["compas"]#["adult", "compas"]

# epsilons for learning quality experiments:
# epsilons = [0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.9825, 0.985, 0.9875, 0.99, 0.9925, 0.995, 0.9975] #0.7, 0.8, 0.9,

epsilons = [0.70, 0.75, 0.80, 0.85, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995] #0.7, 0.8, 0.9,
seeds = []
for i in range(0,100):
    seeds.append(i)
missingSeeds = []
metrics=[1, 3, 4, 5]#, 4, 5]

filteringModes = [0, 1, 2]
# -----------------------------------------------------

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
    dataset = cart_product[expe_id][0]
    epsilon = cart_product[expe_id][1]
    fairnessMetric = cart_product[expe_id][2]
    max_time = cart_product[expe_id][3]
    filteringMode = cart_product[expe_id][4]
    for seed in seeds:
        fileName = './results/%s%s_eps%f_metric%d_LB%d_%s_tLimit%d_single_seed%d%s.csv' %(folderPrefix, dataset, epsilon, fairnessMetric, filteringMode, policy, max_time, seed, archSuffix)
        if not os.path.isfile(fileName):
            missingcnt+=1
            if not expe_id in torerun:
                torerun.append(expe_id)    
            if not seed in missingSeeds:
                missingSeeds.append(seed)   
            runsParams.append([dataset, epsilon, fairnessMetric, max_time, filteringMode, seed])        
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
    print("Seed values that are missing are: ", missingSeeds)
    print("total = %d missing seed values" %len(missingSeeds))
    '''for i in range(nb_expes):
        if not i in torerun:
            print("expe ", i, " ok")'''
print("Found %d files" %cnt)
if missingcnt > 0:
    print("Missing %d files" %missingcnt)


#print(runsParams)
