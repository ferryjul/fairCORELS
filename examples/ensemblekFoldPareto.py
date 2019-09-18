from faircorels import *
from metrics import ConfusionMatrix, Metric
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import sys
import random
import csv

def average(aList):
    nb = 0
    sumTot = 0
    for El in aList:
        nb = nb + 1
        sumTot = sumTot + El
    return sumTot/nb

def runCORELS(X, Y, baggingNB, lambdaV, modeC, epsilon, foldNB, YtestS, testS):
    r_nb = baggingNB #(baggingNB+1)*(baggingNB+1)*10*(foldNB+1)
    #print("--- ENTERING FOLD %d (bag %d seed %lf) ---" %(foldNB,baggingNB, r_nb))
    random.seed(r_nb)
    X_bag_list = []
    y_bag_list = []
    '''if foldNB == 0:
        print(" --------- BEFORE --------- ")
        print(X)
        print(Y)'''
    for ind in range(sizeSamples):
        inst = random.randrange(len(X)-1) #pick a random instance
        inst = inst + 1 # the len(XX-1) and then +1 allows to avoid the 0 value
        X_bag_list.append(X[inst])
        y_bag_list.append(Y[inst])
    X_bag_train = np.array(X_bag_list)
    y_bag_train = np.array(y_bag_list)
    #if(baggingNB == 0):
    #    print(X_bag_train)
    '''if foldNB == 0:
        print(" --------- AFTER --------- ")
        print(X_bag_train)
        print(y_bag_train)'''
    #print("%d instances in initial train, %d in initial test (bagging)" %(len(X),len(Y)))
    #print("%d instances in train, %d in test (bagging)" %(len(X_bag_train),len(y_bag_train)))
    #print("Calling CORELS with parameters : NBNOdes = %d, lambda = %lf, mode = %d" %(NBNodes,lambdaF,modeC))
    #print("random state = %d" %r_nb)
    #print("Will call fairCORELS")
    c = CorelsClassifier(kbest=1, random_state=r_nb, verbosity=[], map_type="prefix", forbidSensAttr=forbidArg, n_iter=NBNodes, c=lambdaV, max_card=1, policy="bfs", bfs_mode=2, useUnfairnessLB=True, fairness=fairnessMetric, maj_pos=UnprotectedIndex+1, min_pos=ProtectedIndex+1, epsilon=epsilon, mode=modeC)
    c.fit(X_bag_train, y_bag_train, performRestarts=0, initNBNodes=46000, geomRReason=1.5, features=features_tot, prediction_name="(credit_rating)")
    # Load unique test set and compute the model's evaluation on it
    print("Fold ", foldNB, " bag nb ", baggingNB, c.rl())
    #print("Hey Train accuracy {}".format(c.score(X_bag_train, y_bag_train)))
    #print("Hey Test accuracy {}".format(c.score(testS, YtestS)))
    test = c.predict_with_scores(testS)
    #print("returning")
    return test

def performKFold(foldID, epsGlob, modeGlob):
    startTest = int(foldID*setSize)
    endTest =  int((foldID+1)*setSize)
    startTrain1 = int(0)
    endTrain1 = int(foldID*setSize - 1)
    startTrain2 = int((foldID+1)*setSize + 1)
    endTrain2 = int(len(X_tot))
    #print("fold test ", foldID, " : instances [", startTest, ", ", endTest, "]")
    #print("fold train ", foldID, " : instances [", startTrain1, ", ", endTrain1, "] and [", startTrain2, ", ", endTrain2, "]")
    X_trainFold1 = X_tot[startTrain1:np.max([endTrain1,0]),:]
    y_trainFold1 = y_tot[startTrain1:np.max([endTrain1,0])]
    X_trainFold2 = X_tot[startTrain2:np.min([endTrain2,len(X_tot)-1]),:]
    y_trainFold2 = y_tot[startTrain2:np.min([endTrain2,len(X_tot)-1])]
    X_fold_train = np.concatenate((X_trainFold1, X_trainFold2), axis=0)
    y_fold_train = np.concatenate((y_trainFold1, y_trainFold2), axis=0)
    X_fold_test = X_tot[startTest:endTest,:]
    y_fold_test = y_tot[startTest:endTest]
    #print("%d instances in test set, %d instances in train set" %(len(X_fold_test),len(X_fold_train)))
    #print("%d instances in test set, %d instances in train set" %(len(y_fold_test),len(y_fold_train)))
    baggRes = [] # List of (pred, score) arrays
    eps = epsGlob
    mode = modeGlob
    baggRes = Parallel(n_jobs=12)(delayed(runCORELS)(X=X_fold_train, Y=y_fold_train, baggingNB=b, lambdaV=lambdaF, modeC=mode,epsilon=eps, foldNB=foldID, testS=X_fold_test, YtestS=y_fold_test) for b in range(nbSamples))
    #print("%d models computed" %len(baggRes))
    # Compute aggregation prediction
    #print("size of test preds : %d" %len(baggRes[0][0]))
    finalPreds = []
    for i in range(len(y_fold_test)):
        zeroVotes = 0
        oneVotes = 0
        for b in range(len(baggRes)):
            #print("%lf,%lf" %(baggRes[b][0][i],baggRes[b][1][i]))
            if baggRes[b][0][i] == 0:
                zeroVotes = zeroVotes + 1#baggRes[b][1][i]
            else:
                oneVotes = oneVotes + 1#baggRes[b][1][i]
        #print("votes 0 : %lf, votes 1 : %lf" %(zeroVotes, oneVotes))
        if zeroVotes >= oneVotes:
            finalPreds.append(0)
        else:
            finalPreds.append(1)
    finalPreds_bis = []
    for i in range(len(y_fold_test)):
        zeroVotes = 0
        oneVotes = 0
        for b in range(len(baggRes)):
            #print("%lf,%lf" %(baggRes[b][0][i],baggRes[b][1][i]))
            if baggRes[b][0][i] == 0:
                zeroVotes = zeroVotes + baggRes[b][1][i]
            else:
                oneVotes = oneVotes + baggRes[b][1][i]
        #print("votes 0 : %lf, votes 1 : %lf" %(zeroVotes, oneVotes))
        if zeroVotes >= oneVotes:
            finalPreds_bis.append(0)
        else:
            finalPreds_bis.append(1)
    cmTest = ConfusionMatrix(X_fold_test[:,ProtectedIndex], X_fold_test[:,UnprotectedIndex], np.array(finalPreds), y_fold_test)
    cm_minorityTest, cm_majorityTest = cmTest.get_matrix()
    fmTest = Metric(cm_minorityTest, cm_majorityTest)
    acc = 0
    err = 0
    for i in range(len(y_fold_test)):
        if y_fold_test[i] == finalPreds[i]:
            acc = acc + 1
        else:
            err = err + 1
    acc = (float) (acc / len(y_fold_test))
    err = (float) (err / len(y_fold_test))
    cmTest_bis = ConfusionMatrix(X_fold_test[:,ProtectedIndex], X_fold_test[:,UnprotectedIndex], np.array(finalPreds_bis), y_fold_test)
    cm_minorityTest_bis, cm_majorityTest_bis = cmTest_bis.get_matrix()
    fmTest_bis = Metric(cm_minorityTest_bis, cm_majorityTest_bis)
    acc_bis = 0
    err_bis = 0
    for i in range(len(y_fold_test)):
        if y_fold_test[i] == finalPreds_bis[i]:
            acc_bis = acc_bis + 1
        else:
            err_bis = err_bis + 1
    acc_bis = (float) (acc_bis / len(y_fold_test))
    err_bis = (float) (err_bis / len(y_fold_test))
    
    return [[acc, fmTest.statistical_parity(), fmTest.predictive_parity(), fmTest.predictive_equality(), fmTest.equal_opportunity()],[acc_bis, fmTest_bis.statistical_parity(), fmTest_bis.predictive_parity(), fmTest_bis.predictive_equality(), fmTest_bis.equal_opportunity()]]

dataset_name = sys.argv[3]
fairnessMetric = int(sys.argv[1])
NBNodes = 15000
nbSamples = 24 # Bootstrap sampling built sets
relativeSizeSamples = 0.9
lambdaF = 0.0001
kFold = 10 # Enter here the number of folds for the k-fold cross-validation
NBPoints = 5
if(int(sys.argv[2]) == 0):
    forbidArg = False
    fStr = "sens_arg"
elif(int(sys.argv[2]) == 1):
    forbidArg = True
    fStr = "no_sens_arg"
if dataset_name == "Adult":
    X_tot, y_tot, features_tot, prediction_tot = load_from_csv("data/adult_full_binary.csv")
    prediction_name_ = "(income)"
    UnprotectedIndex = 19
    ProtectedIndex = 18
elif dataset_name == "German_credit":
    X_tot, y_tot, features_tot, prediction_tot = load_from_csv("data/german_credit_binary.csv")
    UnprotectedIndex = -1
    ProtectedIndex = 47
    prediction_name_ = "(credit_rating)"
elif dataset_name == "Compas":
    X_tot, y_tot, features_tot, prediction_tot = load_from_csv("data/compas_full_binary.csv")
    UnprotectedIndex = 9
    ProtectedIndex = 7
    prediction_name_ = "(two_year_recid)"
sizeSamples = int(relativeSizeSamples * (((kFold-1)*len(X_tot))/kFold))
print("size samples : %d" %sizeSamples)
print("--- DATASET INFO --- ")
print("Nombre d'attributs : %d, Nombre d'instances : %d" % (len(features_tot),len(X_tot)))
print("Prediction : %s" %prediction_tot)
print("--------------------")

print("Will perform %d-folds cross-validation" %kFold)
setSize = (len(X_tot)) / kFold
print("Fold size = %d instances" %setSize)


accuracy_list_test = []
statistical_parity_list_test = []
predictive_parity_list_test = []
predictive_equality_list_test = []
equal_opportunity_list_test = []

accuracy_list_test_bis = []
statistical_parity_list_test_bis = []
predictive_parity_list_test_bis = []
predictive_equality_list_test_bis = []
equal_opportunity_list_test_bis = []

accuracy_list_test_temp = []
statistical_parity_list_test_temp = []
predictive_parity_list_test_temp = []
predictive_equality_list_test_temp = []
equal_opportunity_list_test_temp = []

accuracy_list_test_bis_temp = []
statistical_parity_list_test_bis_temp = []
predictive_parity_list_test_bis_temp = []
predictive_equality_list_test_bis_temp = []
equal_opportunity_list_test_bis_temp = []

# Max fairness
returnList = Parallel(n_jobs=1)(delayed(performKFold)(foldID=_foldID, modeGlob=3, epsGlob=1) for _foldID in range(kFold))
for aReturn in returnList:
    accuracy_list_test_temp.append(aReturn[0][0])
    statistical_parity_list_test_temp.append(aReturn[0][1])
    predictive_parity_list_test_temp.append(aReturn[0][2])
    predictive_equality_list_test_temp.append(aReturn[0][3])
    equal_opportunity_list_test_temp.append(aReturn[0][4])
    accuracy_list_test_bis_temp.append(aReturn[1][0])
    statistical_parity_list_test_bis_temp.append(aReturn[1][1])
    predictive_parity_list_test_bis_temp.append(aReturn[1][2])
    predictive_equality_list_test_bis_temp.append(aReturn[1][3])
    equal_opportunity_list_test_bis_temp.append(aReturn[1][4])
accuracy_list_test.append(average(accuracy_list_test_temp))
statistical_parity_list_test.append(average(statistical_parity_list_test_temp))
predictive_parity_list_test.append(average(predictive_parity_list_test_temp))
predictive_equality_list_test.append(average(predictive_equality_list_test_temp))
equal_opportunity_list_test.append(average(equal_opportunity_list_test_temp))
accuracy_list_test_bis.append(average(accuracy_list_test_bis_temp))
statistical_parity_list_test_bis.append(average(statistical_parity_list_test_bis_temp))
predictive_parity_list_test_bis.append(average(predictive_parity_list_test_bis_temp))
predictive_equality_list_test_bis.append(average(predictive_equality_list_test_bis_temp))
equal_opportunity_list_test_bis.append(average(equal_opportunity_list_test_bis_temp))
# Max acc
returnList = Parallel(n_jobs=1)(delayed(performKFold)(foldID=_foldID, modeGlob=3, epsGlob=0) for _foldID in range(kFold))
for aReturn in returnList:
    accuracy_list_test_temp.append(aReturn[0][0])
    statistical_parity_list_test_temp.append(aReturn[0][1])
    predictive_parity_list_test_temp.append(aReturn[0][2])
    predictive_equality_list_test_temp.append(aReturn[0][3])
    equal_opportunity_list_test_temp.append(aReturn[0][4])
    accuracy_list_test_bis_temp.append(aReturn[1][0])
    statistical_parity_list_test_bis_temp.append(aReturn[1][1])
    predictive_parity_list_test_bis_temp.append(aReturn[1][2])
    predictive_equality_list_test_bis_temp.append(aReturn[1][3])
    equal_opportunity_list_test_bis_temp.append(aReturn[1][4])
accuracy_list_test.append(average(accuracy_list_test_temp))
statistical_parity_list_test.append(average(statistical_parity_list_test_temp))
predictive_parity_list_test.append(average(predictive_parity_list_test_temp))
predictive_equality_list_test.append(average(predictive_equality_list_test_temp))
equal_opportunity_list_test.append(average(equal_opportunity_list_test_temp))
accuracy_list_test_bis.append(average(accuracy_list_test_bis_temp))
statistical_parity_list_test_bis.append(average(statistical_parity_list_test_bis_temp))
predictive_parity_list_test_bis.append(average(predictive_parity_list_test_bis_temp))
predictive_equality_list_test_bis.append(average(predictive_equality_list_test_bis_temp))
equal_opportunity_list_test_bis.append(average(equal_opportunity_list_test_bis_temp))

if(fairnessMetric == 1):
    minF = statistical_parity_list_test[1]
    maxF = statistical_parity_list_test[0]
elif(fairnessMetric == 2):
    minF = predictive_parity_list_test[1]
    maxF = predictive_parity_list_test[0]
elif(fairnessMetric == 3):
    minF = predictive_equality_list_test[1]
    maxF = predictive_equality_list_test[0]
elif(fairnessMetric == 4):
    minF = equal_opportunity_list_test[1]
    maxF = equal_opportunity_list_test[0]
minF = 1 - minF
maxF = 1 - maxF
delta = abs(minF - maxF)/NBPoints
epsList = [1,0]
eps = minF + delta
print("minF:", minF, "delta : ", delta)
while eps < maxF:
    epsList.append(eps)
    returnList = Parallel(n_jobs=1)(delayed(performKFold)(foldID=_foldID, modeGlob=3, epsGlob=eps) for _foldID in range(kFold))
    for aReturn in returnList:
        accuracy_list_test_temp.append(aReturn[0][0])
        statistical_parity_list_test_temp.append(aReturn[0][1])
        predictive_parity_list_test_temp.append(aReturn[0][2])
        predictive_equality_list_test_temp.append(aReturn[0][3])
        equal_opportunity_list_test_temp.append(aReturn[0][4])
        accuracy_list_test_bis_temp.append(aReturn[1][0])
        statistical_parity_list_test_bis_temp.append(aReturn[1][1])
        predictive_parity_list_test_bis_temp.append(aReturn[1][2])
        predictive_equality_list_test_bis_temp.append(aReturn[1][3])
        equal_opportunity_list_test_bis_temp.append(aReturn[1][4])
    accuracy_list_test.append(average(accuracy_list_test_temp))
    statistical_parity_list_test.append(average(statistical_parity_list_test_temp))
    predictive_parity_list_test.append(average(predictive_parity_list_test_temp))
    predictive_equality_list_test.append(average(predictive_equality_list_test_temp))
    equal_opportunity_list_test.append(average(equal_opportunity_list_test_temp))
    accuracy_list_test_bis.append(average(accuracy_list_test_bis_temp))
    statistical_parity_list_test_bis.append(average(statistical_parity_list_test_bis_temp))
    predictive_parity_list_test_bis.append(average(predictive_parity_list_test_bis_temp))
    predictive_equality_list_test_bis.append(average(predictive_equality_list_test_bis_temp))
    equal_opportunity_list_test_bis.append(average(equal_opportunity_list_test_bis_temp))
    eps = eps + delta
with open('./testsEnsembleNoAware/results_%s_%d_%s_debug.csv' %(fStr,fairnessMetric,dataset_name), mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['(Lambda = %lf)' %lambdaF, 'FairnessMetric = %d' %fairnessMetric, '%d models' %nbSamples, '%d max nodes in trie' %NBNodes])
    csv_writer.writerow(['Epsilon', 'Accuracy (aggr 1)', 'Statistical parity (aggr 1)', 
    'Predictive Parity (aggr 1)', 'Predictive Equality (aggr 1)', 
    'Equal Opportunity (aggr 1)', 'Accuracy (aggr 2)', 
    'Statistical parity (aggr 2)', 'Predictive Parity (aggr 2)', 
    'Predictive Equality (aggr 2)', 'Equal Opportunity (aggr 2)'])
    index = 0
    print("lengths  should all be equal :") # Debug
    print(len(epsList))
    print(len(accuracy_list_test))
    print(len(statistical_parity_list_test))
    print(len(predictive_parity_list_test))
    print(len(predictive_equality_list_test))
    print(len(equal_opportunity_list_test_bis))
    print(len(accuracy_list_test_bis))
    print(len(statistical_parity_list_test_bis))
    for i in range(len(epsList)):
        #print("index = %d, i = %d\n" %(index,i))
        csv_writer.writerow([epsList[i], 
        accuracy_list_test[i], 
        statistical_parity_list_test[i], 
        predictive_parity_list_test[i], 
        predictive_equality_list_test[i],
        equal_opportunity_list_test_bis[i],
        accuracy_list_test_bis[i], 
        statistical_parity_list_test_bis[i], 
        predictive_parity_list_test_bis[i], 
        predictive_equality_list_test_bis[i],
        equal_opportunity_list_test_bis[i]])
    index = index + 1
    csv_writer.writerow(["", "", "", "", "", "", "", "", "", "", ""])
'''
#print("-------------------------- Learned rulelist ------------------------------")
print("------------------ SIMPLE VOTE AGGREG : ------------------")
print("accuracy list : ", accuracy_list_test)
print("Accuracy %lf" %average(accuracy_list_test))
print("=========> Statistical parity %lf" %average(statistical_parity_list_test))
print("=========> Predictive parity %lf" %average(predictive_parity_list_test))
print("=========> Predictive equality %lf" %average(predictive_equality_list_test))
print("=========> Equal opportunity %lf" %average(equal_opportunity_list_test))
print("------------------ SCORES-BASED VOTE AGGREG : ------------------")
print("accuracy list : ", accuracy_list_test_bis)
print("Accuracy %lf" %average(accuracy_list_test_bis))
print("=========> Statistical parity %lf" %average(statistical_parity_list_test_bis))
print("=========> Predictive parity %lf" %average(predictive_parity_list_test_bis))
print("=========> Predictive equality %lf" %average(predictive_equality_list_test_bis))
print("=========> Equal opportunity %lf" %average(equal_opportunity_list_test_bis))
'''