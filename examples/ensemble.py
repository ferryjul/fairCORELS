from faircorels import *
import matplotlib.pyplot as plt
import csv
import math
import random
from metrics import ConfusionMatrix, Metric
from joblib import Parallel, delayed
import sys
import os
import numpy as np

def computeSTD(aList):
    variance = 0
    moyenne = average(aList)
    for aVal in aList:
        variance = variance + ((aVal - moyenne)*(aVal - moyenne))
    variance = variance / len(aList)
    return math.sqrt(variance)

def average(aList):
    nb = 0
    sumTot = 0
    for El in aList:
        nb = nb + 1
        sumTot = sumTot + El
    return sumTot/nb

def build_bootstrap_body(_origX, _origY, _origF, _origP, sampleID, size):
    newDatasetName = "_tmp_%d/adult_full_binary_%d.csv" %(fairnessMetric, sampleID)
    # First generate list of instances indices
    random.seed(sampleID*10)
    indicesList = []
    indicesList.append(0) #Will allow the names' writting first
    for ind in range(size):
        inst = random.randrange(len(_origX)-1) #pick a random instance
        inst = inst + 1 # the len(XX-1) and then +1 allows to avoid the 0 value
        indicesList.append(inst)
    with open(newDatasetName, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        heads = _origF
        heads.append(_origP)
        csv_writer.writerow(heads)
        for index in indicesList:
            inst = _origX[index].tolist()
            inst.append(_origY[index])
            csv_writer.writerow(inst)
    print("Bootstrap set %d built and saved." %sampleID)

def build_bootstrap(origX, origY, origF, origP, nbSamples, sizeSamples):
    print("Bootstrap sampling begins for %d sets of %d instances." %(nbSamples, sizeSamples))
    Parallel(n_jobs=-1)(delayed(build_bootstrap_body)(_origX=origX, _origY=origY, _origF=origF, _origP=origP, sampleID = i, size = sizeSamples) for i in range(nbSamples))
    
def runCORELS(baggingNB, lambdaV, modeC, epsilon):
    X_fold_train, y_fold_train, features_tot, prediction_tot = load_from_csv("_tmp_%d/adult_full_binary_%d.csv" % (fairnessMetric, baggingNB))
    c = CorelsClassifier(verbosity=[],map_type="prefix", n_iter=NBNodes, c=lambdaV, max_card=1, policy="bfs", bfs_mode=2, useUnfairnessLB=True, fairness=fairnessMetric, maj_pos=UnprotectedIndex+1, min_pos=ProtectedIndex+1, epsilon=epsilon, mode=modeC)
    c.fit(X_fold_train, y_fold_train, features=features_tot, prediction_name="(credit_rating)")
    # Load unique test set and compute the model's evaluation on it
    # print(c.rl())
    return c.predict_with_scores(X_test)

def clean_lists(s1, s2, s3, s4): # Removes repetitions and dominated solutions
    s1Tmp = []
    s2Tmp = []
    s3Tmp = []
    s4Tmp = []
    for i in range(len(s1)):
        isDominated = False
        repetition = False
        for j in range(len(s1)):
            if i != j:
                if ((s1[i]<s1[j]) and (s2[i][fairnessMetric-1]>=s2[j][fairnessMetric-1])) or ((s2[i][fairnessMetric-1]>s2[j][fairnessMetric-1]) and (s1[i]<=s1[j])):
                    isDominated = True
                    print("[%lf,%lf] dominated by [%lf,%lf]\n" %(s1[i], s2[i][fairnessMetric-1], s1[j], s2[j][fairnessMetric-1]))
                elif (s1[i] == s1[j]) and (s2[i][fairnessMetric-1] == s2[j][fairnessMetric-1] and j < i):
                    repetition = True
        if(not isDominated and not repetition):
            s1Tmp.append(s1[i])
            s2Tmp.append(s2[i])
           # s3Tmp.append(s3[i])
           # s4Tmp.append(s4[i])
    return s1Tmp, s2Tmp, s3Tmp, s4Tmp
    
def buildAndAggreg(l, mode=3, eps=0.0):
    baggRes = [] # List of (pred, score) arrays
    baggRes = Parallel(n_jobs=-1)(delayed(runCORELS)(baggingNB=b, lambdaV=l, modeC=mode,epsilon=eps) for b in range(nbSamples))
    # Compute aggregation prediction
    finalPreds = []
    for i in range(len(y_test)):
        zeroVotes = 0
        oneVotes = 0
        for b in range(nbSamples):
            #print("vote %d; score %lf" %(baggRes[b][0][i], baggRes[b][1][i]))
            if baggRes[b][0][i] == 0:
                zeroVotes = zeroVotes + baggRes[b][1][i]
            else:
                oneVotes = oneVotes + baggRes[b][1][i]
        if zeroVotes >= oneVotes:
            finalPreds.append(0)
        else:
            finalPreds.append(1)
        '''if(zeroVotes > 0 and oneVotes > 0):
            print("For instance 5 : %lf votes 1 %lf votes 0" %(oneVotes, zeroVotes))'''
        '''print("%d models voted" %len(baggRes))'''
    cmTest = ConfusionMatrix(X_test[:,ProtectedIndex], X_test[:,UnprotectedIndex], np.array(finalPreds), y_test)
    cm_minorityTest, cm_majorityTest = cmTest.get_matrix()
    fmTest = Metric(cm_minorityTest, cm_majorityTest)
    acc = 0
    for i in range(len(y_test)):
        if y_test[i] == finalPreds[i]:
            acc = acc + 1
    acc = (float) (acc / len(y_test))
    return [acc, [fmTest.statistical_parity(), fmTest.predictive_parity(), fmTest.predictive_equality(), fmTest.equal_opportunity()]]

#dataset_name = "Adult"
dataset_name = "German_credit"
#kFold = 10 # Enter here the number of folds for the k-fold cross-validation
UnprotectedIndex = 19
ProtectedIndex = 18
NBPoints = 55
fairnessMetric = int(sys.argv[1])
NBNodes = 1500000
nbSamples = 36 # Bootstrap sampling built sets
relativeSizeSamples = 0.9
lambdaF = 0.005
if(dataset_name == "Adult"):
    X_tot, y_tot, features_tot, prediction_tot = load_from_csv("data/adult_train_binary.csv")
    X_test, y_test, features_test, prediction_test = load_from_csv("data/adult_test_binary.csv")
elif dataset_name == "German_credit":
    X_tot, y_tot, features_tot, prediction_tot = load_from_csv("data/german_credit_train_total.csv")
    X_test, y_test, features_test, prediction_test = load_from_csv("data/german_credit_test_total.csv")
sizeSamples = int(relativeSizeSamples * len(X_tot))
'''print("X : ", X_tot)
print("Y :", y_tot)
print("Feats : ", features_tot)
print("Preds : ", prediction_tot)'''
print("--- DATASET INFO --- ")
print("Nombre d'attributs : %d, Nombre d'instances : %d" % (len(features_tot),len(X_tot)))
print("Prediction : %s" %prediction_tot)
print("--------------------")
print("Will perform bagging with %d bootstrap-sampling built sets of %d instances." %(nbSamples, sizeSamples))
build_bootstrap(X_tot, y_tot, features_tot, prediction_tot, nbSamples, sizeSamples)
print("--------------------")
print("Chosen fairness metric : %d" %fairnessMetric)
#print("Will perform %d-folds cross-validation" %kFold)
#setSize = (len(X_tot)) / kFold
#print("Fold size = %d instances" %setSize)
print("--------------------")
accuracy_list_test_tot = []
fairness_list_test_tot = []
accuracySTD_list_test_tot = []
fairnessSTD_list_test_tot = []

# Initial solution for max fairness
results = buildAndAggreg(l=lambdaF, mode=2)
accuracy_list_test_tot.append(results[0])
fairness_list_test_tot.append(results[1])
print("Initial max fairness solution : accuracy =  %lf, unfairness = %lf" %(accuracy_list_test_tot[0], fairness_list_test_tot[0][fairnessMetric-1]))
# Initial solution for max accuracy
results = buildAndAggreg(l=lambdaF, mode=3, eps=0.0)
accuracy_list_test_tot.append(results[0])
fairness_list_test_tot.append(results[1])

print("Initial max accuracy solution : accuracy =  %lf, unfairness = %lf" %(accuracy_list_test_tot[1], fairness_list_test_tot[1][fairnessMetric-1]))

# Loop
dist = fairness_list_test_tot[1][fairnessMetric-1] - fairness_list_test_tot[0][fairnessMetric-1]
print("dist = %lf\n" %dist)
delta = dist/NBPoints
unfairnessLim = fairness_list_test_tot[0][fairnessMetric-1] + delta
ind = 0
while unfairnessLim < fairness_list_test_tot[1][fairnessMetric-1]:
    results = buildAndAggreg(l=lambdaF, mode=3, eps=1-unfairnessLim)
    accuracy_list_test_tot.append(results[0])
    fairness_list_test_tot.append(results[1])
    ind = ind + 1
    print("Solution : accuracy =  %lf, unfairness = %lf" %(accuracy_list_test_tot[ind+1], fairness_list_test_tot[ind+1][fairnessMetric-1]))
    print("------ %d/%d done. ----------- " %(ind,NBPoints-1))
    unfairnessLim = unfairnessLim + delta

print("Eliminating dominated solutions...")
accuracy_list_test_tot, fairness_list_test_tot, accuracySTD_list_test_tot, fairnessSTD_list_test_tot = clean_lists(accuracy_list_test_tot, fairness_list_test_tot, accuracySTD_list_test_tot, fairnessSTD_list_test_tot)
fairness_list_plot = []
for i in range(len(fairness_list_test_tot)):
    fairness_list_plot.append(fairness_list_test_tot[i][fairnessMetric-1])

# Export the results
plt.scatter(accuracy_list_test_tot, fairness_list_plot, label = "Testing, lambda = 0.0005")
# Add title and axis names
plt.title("Pareto Front approximation (dataset = %s,\nProtected attribute = %s Unprotected attribute = %s)" %(dataset_name, ProtectedIndex, UnprotectedIndex))
plt.xlabel('Error')
plt.ylabel('Unfairness (#%d)' %fairnessMetric)
plt.legend(loc='lower left')
plt.axis([0,1,0,1])
plt.autoscale(tight=True)
plt.savefig("./plots/result_plot_fairness_%d_ensemble.png" %fairnessMetric)

with open('./plots/results_fairness_%d_ensemble.csv' %fairnessMetric, mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Lambda', 'Unfairness(%d)' %fairnessMetric, 'Error'])#, 'Fairness STD', 'Accuracy STD'])
    index = 0
    for i in range(len(accuracy_list_test_tot)):
        #print("index = %d, i = %d\n" %(index,i))
        csv_writer.writerow([lambdaF, fairness_list_test_tot[i][fairnessMetric-1], accuracy_list_test_tot[i]])#,fairnessSTD_list_test_tot[i][fairnessMetric-1], accuracySTD_list_test_tot[i]])
    index = index + 1
    csv_writer.writerow(["", "", "", ""])#, "", ""])
# Delete all temporary built datasets
if os.system('rm _tmp_%d/*' %fairnessMetric) == 0:
    print("Deleted all temporary .csv files")
