from faircorels import *
from hyperopt import fmin, tpe, hp
from metrics import ConfusionMatrix, Metric
import pandas as pd
import numpy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import csv
import math

def average(aList):
    nb = 0
    sumTot = 0
    for El in aList:
        nb = nb + 1
        sumTot = sumTot + El
    return sumTot/nb

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
                if ((s1[i]>s1[j]) and (s2[i][fairnessMetric-1]>=s2[j][fairnessMetric-1])) or ((s2[i][fairnessMetric-1]>s2[j][fairnessMetric-1]) and (s1[i]>=s1[j])):
                    isDominated = True
                    print("[%lf,%lf] dominated by [%lf,%lf]\n" %(s1[i], s2[i][fairnessMetric-1], s1[j], s2[j][fairnessMetric-1]))
                elif (s1[i] == s1[j]) and (s2[i][fairnessMetric-1] == s2[j][fairnessMetric-1] and j < i):
                    repetition = True
        if(not isDominated and not repetition):
            s1Tmp.append(s1[i])
            s2Tmp.append(s2[i])
            s3Tmp.append(s3[i])
            s4Tmp.append(s4[i])
    return s1Tmp, s2Tmp, s3Tmp, s4Tmp

def computeSTD(aList):
    variance = 0
    moyenne = average(aList)
    for aVal in aList:
        variance = variance + ((aVal - moyenne)*(aVal - moyenne))
    variance = variance / len(aList)
    return math.sqrt(variance)
    
def performKFold(foldID, lambdaF, minFairness=0.0, modeC=3):
    startTest = int(foldID*setSize)
    endTest =  int((foldID+1)*setSize)
    startTrain1 = int(0)
    endTrain1 = int(foldID*setSize - 1)
    startTrain2 = int((foldID+1)*setSize + 1)
    endTrain2 = int(len(X_tot))
    #print("fold test ", foldID, " : instances [", startTest, ", ", endTest, "]")
    #print("fold train ", foldID, " : instances [", startTrain1, ", ", endTrain1, "] and [", startTrain2, ", ", endTrain2, "]")
    X_trainFold1 = X_tot[startTrain1:numpy.max([endTrain1,0]),:]
    y_trainFold1 = y_tot[startTrain1:numpy.max([endTrain1,0])]
    X_trainFold2 = X_tot[startTrain2:numpy.min([endTrain2,len(X_tot)-1]),:]
    y_trainFold2 = y_tot[startTrain2:numpy.min([endTrain2,len(X_tot)-1])]
    X_fold_train = numpy.concatenate((X_trainFold1, X_trainFold2), axis=0)
    y_fold_train = numpy.concatenate((y_trainFold1, y_trainFold2), axis=0)
    X_fold_test = X_tot[startTest:endTest,:]
    y_fold_test = y_tot[startTest:endTest]
    #print("%d instances in test set, %d instances in train set" %(len(X_fold_test),len(X_fold_train)))
    c = CorelsClassifier(verbosity=[],map_type="prefix", n_iter=NBNodes, c=lambdaF, max_card=1, policy="objective", useUnfairnessLB=True, fairness=fairnessMetric, maj_pos=UnprotectedIndex+1, min_pos=ProtectedIndex+1, epsilon=minFairness, mode=modeC)
    c.fit(X_fold_train, y_fold_train, features=features_tot, prediction_name="(income:>50K)")
    # Accuracy sur le train set
    accuracy_train = (c.score(X_fold_train, y_fold_train))
    # Accuracy sur le test set
    accuracy_test = (c.score(X_fold_test, y_fold_test))
    # Fairness metrics sur le train set
    y_pred_train = c.predict(X_fold_train)
    cmTrain = ConfusionMatrix(X_fold_train[:,ProtectedIndex], X_fold_train[:,UnprotectedIndex], y_pred_train, y_fold_train)
    cm_minorityTrain, cm_majorityTrain = cmTrain.get_matrix()
    fmTrain = Metric(cm_minorityTrain, cm_majorityTrain)
    #print("training set : TPmaj = %d, FPmaj = %d, TNmaj = %d, FNmaj = %d, TPmin = %d, FPmin = %d, TNmin = %d, FNmin = %d" %(cm_majorityTrain['TP'],cm_majorityTrain['FP'],cm_majorityTrain['TN'],cm_majorityTrain['FN'],cm_minorityTrain['TP'],cm_minorityTrain['FP'],cm_minorityTrain['TN'],cm_minorityTrain['FN']))
    # Fairness metrics sur le test set
    y_pred_test = c.predict(X_fold_test)
    cmTest = ConfusionMatrix(X_fold_test[:,ProtectedIndex], X_fold_test[:,UnprotectedIndex], y_pred_test, y_fold_test)
    cm_minorityTest, cm_majorityTest = cmTest.get_matrix()
    fmTest = Metric(cm_minorityTest, cm_majorityTest)
    return [accuracy_train, accuracy_test, fmTrain.statistical_parity(), fmTrain.predictive_parity(), fmTrain.predictive_equality(), fmTrain.equal_opportunity(), fmTest.statistical_parity(), fmTest.predictive_parity(), fmTest.predictive_equality(), fmTest.equal_opportunity()]

def computeObjectiveWithCV(lambdaFactor):
    l = lambdaFactor[0]
    returnList = Parallel(n_jobs=-1)(delayed(performKFold)(foldID=_foldID, minFairness=1-unfairnessLim, lambdaF=l) for _foldID in range(kFold))
    accuracy_list_test = []
    statistical_parity_list_test = []
    predictive_parity_list_test = []
    predictive_equality_list_test = []
    equal_opportunity_list_test = []
    for aReturn in returnList:
        accuracy_list_test.append(aReturn[1])
        statistical_parity_list_test.append(aReturn[6])
        predictive_parity_list_test.append(aReturn[7])
        predictive_equality_list_test.append(aReturn[8])
        equal_opportunity_list_test.append(aReturn[9])
    fairnessM = [average(statistical_parity_list_test),average(predictive_parity_list_test),average(predictive_equality_list_test),average(equal_opportunity_list_test)]
    ret = 1 - average(accuracy_list_test) # Misclassification error
    if unfairnessLim < fairnessM[fairnessMetric-1]: # Too much unfairness -> Penalize the solution
        ret = ret + (fairnessM[fairnessMetric-1]-unfairnessLim)
    return ret

def runHyperOpt():
    bestLam = fmin(fn=computeObjectiveWithCV,
    space=[hp.uniform('lambdaFactor', 0.00001, 0.05)],
    algo=tpe.suggest,
    max_evals=max_evals)
    return bestLam

kFold = 10 # Enter here the number of folds for the k-fold cross-validation
UnprotectedIndex = 19
ProtectedIndex = 18
NBPoints = 15
fairnessMetric = 3
NBNodes = 300000
max_evals = 20
dataset_name = "Adult"
X_tot, y_tot, features_tot, prediction_tot = load_from_csv("data/adult_full_binary.csv")
print("--- DATASET INFO --- ")
print("Nombre d'attributs : %d, Nombre d'instances : %d" % (len(features_tot),len(X_tot)))
print("Prediction : %s" %prediction_tot)
print("--------------------")
print("Chosen fairness metric : %d" %fairnessMetric)
print("Will perform %d-folds cross-validation" %kFold)
setSize = (len(X_tot)) / kFold
print("Fold size = %d instances" %setSize)

accuracy_list_test_tot = []
fairness_list_test_tot = []
accuracySTD_list_test_tot = []
fairnessSTD_list_test_tot = []
accuracy_list_train_tot = []
fairness_list_train_tot = []
accuracySTD_list_train_tot = []
fairnessSTD_list_train_tot = []
# Initial solution for max fairness
# Useless to call Hyperopt here, the dumb solution will always be found in mode 2.
returnList = Parallel(n_jobs=-1)(delayed(performKFold)(foldID=_foldID, modeC=2, lambdaF=0.001) for _foldID in range(kFold))
    
accuracy_list_test = []
statistical_parity_list_test = []
predictive_parity_list_test = []
predictive_equality_list_test = []
equal_opportunity_list_test = []

accuracy_list_train = []
statistical_parity_list_train = []
predictive_parity_list_train = []
predictive_equality_list_train = []
equal_opportunity_list_train = []

for aReturn in returnList:
    accuracy_list_train.append(aReturn[0])
    accuracy_list_test.append(aReturn[1])
    statistical_parity_list_train.append(aReturn[2])
    predictive_parity_list_train.append(aReturn[3])
    predictive_equality_list_train.append(aReturn[4])
    equal_opportunity_list_train.append(aReturn[5])
    statistical_parity_list_test.append(aReturn[6])
    predictive_parity_list_test.append(aReturn[7])
    predictive_equality_list_test.append(aReturn[8])
    equal_opportunity_list_test.append(aReturn[9])
print("--- Initial solution for max fairness: ---")
print("----------- Training set metrics : ----------- ")
print("Accuracy: %lf" %average(accuracy_list_train))
print("=========> Statistical parity %lf" %average(statistical_parity_list_train))
print("=========> Predictive parity %lf" %average(predictive_parity_list_train))
print("=========> Predictive equality %lf" %average(predictive_equality_list_train))
print("=========> Equal opportunity %lf" %average(equal_opportunity_list_train))
print("----------- Test set metrics : --------------- ")
print("Accuracy %lf" %average(accuracy_list_test))
print("=========> Statistical parity %lf" %average(statistical_parity_list_test))
print("=========> Predictive parity %lf" %average(predictive_parity_list_test))
print("=========> Predictive equality %lf" %average(predictive_equality_list_test))
print("=========> Equal opportunity %lf" %average(equal_opportunity_list_test))
accuracy_list_test_tot.append(1-average(accuracy_list_test))
fairness_list_test_tot.append([average(statistical_parity_list_test),average(predictive_parity_list_test),average(predictive_equality_list_test),average(equal_opportunity_list_test)])
accuracySTD_list_test_tot.append(computeSTD(accuracy_list_test))
fairnessSTD_list_test_tot.append([computeSTD(statistical_parity_list_test),computeSTD(predictive_parity_list_test),computeSTD(predictive_equality_list_test),computeSTD(equal_opportunity_list_test)])
accuracy_list_train_tot.append(1-average(accuracy_list_train))
fairness_list_train_tot.append([average(statistical_parity_list_train),average(predictive_parity_list_train),average(predictive_equality_list_train),average(equal_opportunity_list_train)])
accuracySTD_list_train_tot.append(computeSTD(accuracy_list_train))
fairnessSTD_list_train_tot.append([computeSTD(statistical_parity_list_train),computeSTD(predictive_parity_list_train),computeSTD(predictive_equality_list_train),computeSTD(equal_opportunity_list_train)])


# Initial solution for max accuracy
unfairnessLim = 1.0 
bestLambda = runHyperOpt().get('lambdaFactor')
returnList = Parallel(n_jobs=-1)(delayed(performKFold)(foldID=_foldID, minFairness=0.0, lambdaF=bestLambda) for _foldID in range(kFold))
accuracy_list_test = []
statistical_parity_list_test = []
predictive_parity_list_test = []
predictive_equality_list_test = []
equal_opportunity_list_test = []

accuracy_list_train = []
statistical_parity_list_train = []
predictive_parity_list_train = []
predictive_equality_list_train = []
equal_opportunity_list_train = []
print("--- Initial solution for max accuracy : ---")
for aReturn in returnList:
    accuracy_list_train.append(aReturn[0])
    accuracy_list_test.append(aReturn[1])
    statistical_parity_list_train.append(aReturn[2])
    predictive_parity_list_train.append(aReturn[3])
    predictive_equality_list_train.append(aReturn[4])
    equal_opportunity_list_train.append(aReturn[5])
    statistical_parity_list_test.append(aReturn[6])
    predictive_parity_list_test.append(aReturn[7])
    predictive_equality_list_test.append(aReturn[8])
    equal_opportunity_list_test.append(aReturn[9])
print("----------- Training set metrics : ----------- ")
print("Accuracy: %lf" %average(accuracy_list_train))
print("=========> Statistical parity %lf" %average(statistical_parity_list_train))
print("=========> Predictive parity %lf" %average(predictive_parity_list_train))
print("=========> Predictive equality %lf" %average(predictive_equality_list_train))
print("=========> Equal opportunity %lf" %average(equal_opportunity_list_train))
print("----------- Test set metrics : --------------- ")
print("Accuracy %lf" %average(accuracy_list_test))
print("=========> Statistical parity %lf" %average(statistical_parity_list_test))
print("=========> Predictive parity %lf" %average(predictive_parity_list_test))
print("=========> Predictive equality %lf" %average(predictive_equality_list_test))
print("=========> Equal opportunity %lf" %average(equal_opportunity_list_test))
accuracy_list_test_tot.append(1-average(accuracy_list_test))
fairness_list_test_tot.append([average(statistical_parity_list_test),average(predictive_parity_list_test),average(predictive_equality_list_test),average(equal_opportunity_list_test)])
accuracySTD_list_test_tot.append(computeSTD(accuracy_list_test))
fairnessSTD_list_test_tot.append([computeSTD(statistical_parity_list_test),computeSTD(predictive_parity_list_test),computeSTD(predictive_equality_list_test),computeSTD(equal_opportunity_list_test)])
accuracy_list_train_tot.append(1-average(accuracy_list_train))
fairness_list_train_tot.append([average(statistical_parity_list_train),average(predictive_parity_list_train),average(predictive_equality_list_train),average(equal_opportunity_list_train)])
accuracySTD_list_train_tot.append(computeSTD(accuracy_list_train))
fairnessSTD_list_train_tot.append([computeSTD(statistical_parity_list_train),computeSTD(predictive_parity_list_train),computeSTD(predictive_equality_list_train),computeSTD(equal_opportunity_list_train)])

dist = fairness_list_train_tot[1][fairnessMetric-1] - fairness_list_train_tot[0][fairnessMetric-1]
print("dist = %lf\n" %dist)
delta = dist/NBPoints
unfairnessLim = fairness_list_train_tot[0][fairnessMetric-1] + delta
while unfairnessLim < fairness_list_train_tot[1][fairnessMetric-1]:
    print("--- Current (1-epsilon) = %lf --- " %(unfairnessLim))
    bestLambda = runHyperOpt().get('lambdaFactor')
    returnList = Parallel(n_jobs=-1)(delayed(performKFold)(foldID=_foldID, minFairness=1-unfairnessLim, lambdaF=bestLambda) for _foldID in range(kFold))
    accuracy_list_test = []
    statistical_parity_list_test = []
    predictive_parity_list_test = []
    predictive_equality_list_test = []
    equal_opportunity_list_test = []
    accuracy_list_train = []
    statistical_parity_list_train = []
    predictive_parity_list_train = []
    predictive_equality_list_train = []
    equal_opportunity_list_train = []
    for aReturn in returnList:
        accuracy_list_train.append(aReturn[0])
        accuracy_list_test.append(aReturn[1])
        statistical_parity_list_train.append(aReturn[2])
        predictive_parity_list_train.append(aReturn[3])
        predictive_equality_list_train.append(aReturn[4])
        equal_opportunity_list_train.append(aReturn[5])
        statistical_parity_list_test.append(aReturn[6])
        predictive_parity_list_test.append(aReturn[7])
        predictive_equality_list_test.append(aReturn[8])
        equal_opportunity_list_test.append(aReturn[9])
    print("----------- Training set metrics : ----------- ")
    print("Accuracy: %lf" %average(accuracy_list_train))
    print("=========> Statistical parity %lf" %average(statistical_parity_list_train))
    print("=========> Predictive parity %lf" %average(predictive_parity_list_train))
    print("=========> Predictive equality %lf" %average(predictive_equality_list_train))
    print("=========> Equal opportunity %lf" %average(equal_opportunity_list_train))
    print("----------- Test set metrics : --------------- ")
    print("Accuracy %lf" %average(accuracy_list_test))
    print("=========> Statistical parity %lf" %average(statistical_parity_list_test))
    print("=========> Predictive parity %lf" %average(predictive_parity_list_test))
    print("=========> Predictive equality %lf" %average(predictive_equality_list_test))
    print("=========> Equal opportunity %lf" %average(equal_opportunity_list_test))
    accuracy_list_test_tot.append(1-average(accuracy_list_test))
    fairness_list_test_tot.append([average(statistical_parity_list_test),average(predictive_parity_list_test),average(predictive_equality_list_test),average(equal_opportunity_list_test)])
    accuracySTD_list_test_tot.append(computeSTD(accuracy_list_test))
    fairnessSTD_list_test_tot.append([computeSTD(statistical_parity_list_test),computeSTD(predictive_parity_list_test),computeSTD(predictive_equality_list_test),computeSTD(equal_opportunity_list_test)])
    accuracy_list_train_tot.append(1-average(accuracy_list_train))
    fairness_list_train_tot.append([average(statistical_parity_list_train),average(predictive_parity_list_train),average(predictive_equality_list_train),average(equal_opportunity_list_train)])
    accuracySTD_list_train_tot.append(computeSTD(accuracy_list_train))
    fairnessSTD_list_train_tot.append([computeSTD(statistical_parity_list_train),computeSTD(predictive_parity_list_train),computeSTD(predictive_equality_list_train),computeSTD(equal_opportunity_list_train)])
    unfairnessLim = unfairnessLim + delta

print("Eliminating dominated solutions...")
accuracy_list_test_tot, fairness_list_test_tot, accuracySTD_list_test_tot, fairnessSTD_list_test_tot = clean_lists(accuracy_list_test_tot, fairness_list_test_tot, accuracySTD_list_test_tot, fairnessSTD_list_test_tot)
fairness_list_plot = []
for i in range(len(fairness_list_test_tot)):
    fairness_list_plot.append(fairness_list_test_tot[i][fairnessMetric-1])

plt.scatter(accuracy_list_test_tot, fairness_list_plot, label = "Testing, lambda = 0.0005")
# Add title and axis names
plt.title("Pareto Front approximation (dataset = %s,\nProtected attribute = %s Unprotected attribute = %s)" %(dataset_name, ProtectedIndex, UnprotectedIndex))
plt.xlabel('Error')
plt.ylabel('Unfairness (#%d)' %fairnessMetric)
plt.legend(loc='lower left')
plt.axis([0,1,0,1])
plt.autoscale(tight=True)
plt.savefig("./plots/result_plot_fairness_%d.png" %fairnessMetric)

with open('./plots/results_fairness_%d.csv' %fairnessMetric, mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Lambda', 'Unfairness(%d)' %fairnessMetric, 'Error', 'UnfairnessSTD', 'ErrorSTD'])
    index = 0
    for i in range(len(accuracy_list_test_tot)):
        #print("index = %d, i = %d\n" %(index,i))
        csv_writer.writerow([0.0005, fairness_list_test_tot[i][fairnessMetric-1], accuracy_list_test_tot[i], fairnessSTD_list_test_tot[i][fairnessMetric-1], accuracySTD_list_test_tot[i]])
    index = index + 1
    csv_writer.writerow(["", "", "", "", "", ""])
#print("-------------------------- Learned rulelist ------------------------------")
#print(c.rl())

