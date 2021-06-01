#import pandas as pd
import os.path

max_times=[120, 300, 400, 500, 600, 900, 1200, 1800, 2400] 

cart_product = []
policy = "bfs"
folderPrefix= "result_v1_german_broadwell/results_same_arch_4Go_german/" #"results_run_broadwell/"#"results-4Go/" #"results-2.5Go/"
archSuffix = "_broadwell"
# -----------------------------------------------------
datasets= ["german_credit"]#["adult", "compas"]

epsilons = [0.90, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]
seeds = []
for i in range(0,100):
    seeds.append(i)

metrics=[1, 3, 4, 5]#, 4, 5]

filteringModes = [0, 1, 2]
# -----------------------------------------------------
missingFiles = 0
runsParams = []
for d in datasets:
    for e in epsilons:
        #for s in seeds:
        for m in metrics:
            for mt in max_times:
                for fm in filteringModes:
                    cart_product.append([d,e,m,mt,fm])
nb_expes=len(cart_product)
expe_ids = []
missingSeeds = []
for i in range(nb_expes):
    expe_ids.append(i)
print("Needed to run %d expes." %nb_expes)
cnt = 0
torerun = []
for expe_id in expe_ids:
    dataset = cart_product[expe_id][0]
    epsilon = cart_product[expe_id][1]
    fairnessMetric = cart_product[expe_id][2]
    max_time = cart_product[expe_id][3]
    filteringMode = cart_product[expe_id][4]
    for seed in seeds:
        fileName = './results/%s%s_eps%f_metric%d_LB%d_%s_tLimit%d_single_seed%d%s.csv' %(folderPrefix, dataset, epsilon, fairnessMetric, filteringMode, policy, max_time, seed, archSuffix)
        if not os.path.isfile(fileName):
            #print("Missing file: ", fileName, " (expe %d)" %expe_id)
            missingFiles += 1
            if not seed in missingSeeds:
                missingSeeds.append(seed)
            runsParams.append([dataset, epsilon, fairnessMetric, max_time, filteringMode, seed])
            if not expe_id in torerun:
                torerun.append(expe_id)               
        else:
            cnt+=1
            
        '''try:
            fileContent = pd.read_csv(fileName)
        except FileNotFoundError as not_found:
            #print("Missing seed %d, max_time=%d, metric= %d, epsilon=%lf, expe_id=%d" %(seed, max_time, fairnessMetric, epsilon, expe_id))
            if not expe_id in torerun:
                torerun.append(expe_id)'''
        
missingSeeds.sort()
if len(torerun) == 0:
    print("All files found.")
else:
    print("Expes to re-run are: ")
    # print without white space so that copy-paste is easy
    print(str(torerun).replace(" ", ""))
    print("total = %d expes to re-run" %len(torerun), "(missing %d files)" %missingFiles)
    '''for i in range(nb_expes):
        if not i in torerun:
            print("expe ", i, " ok")'''
    print("There are %d missing seeds: " %len(missingSeeds), missingSeeds)
print("Found %d files" %cnt)
#print(runsParams)
