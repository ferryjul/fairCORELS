from faircorels import *
from metrics import ConfusionMatrix, Metric
import pandas as pd
import numpy
from joblib import Parallel, delayed

def average(aList):
    nb = 0
    sumTot = 0
    for El in aList:
        nb = nb + 1
        sumTot = sumTot + El
    return sumTot/nb

def performKFold(foldID):
    startTest = int(foldID*setSize)
    endTest =  int((foldID+1)*setSize)
    startTrain1 = int(0)
    endTrain1 = int(foldID*setSize - 1)
    startTrain2 = int((foldID+1)*setSize + 1)
    endTrain2 = int(len(X_tot))
    print("fold test ", foldID, " : instances [", startTest, ", ", endTest, "]")
    print("fold train ", foldID, " : instances [", startTrain1, ", ", endTrain1, "] and [", startTrain2, ", ", endTrain2, "]")
    X_trainFold1 = X_tot[startTrain1:numpy.max([endTrain1,0]),:]
    y_trainFold1 = y_tot[startTrain1:numpy.max([endTrain1,0])]
    X_trainFold2 = X_tot[startTrain2:numpy.min([endTrain2,len(X_tot)-1]),:]
    y_trainFold2 = y_tot[startTrain2:numpy.min([endTrain2,len(X_tot)-1])]
    X_fold_train = numpy.concatenate((X_trainFold1, X_trainFold2), axis=0)
    y_fold_train = numpy.concatenate((y_trainFold1, y_trainFold2), axis=0)
    X_fold_test = X_tot[startTest:endTest,:]
    y_fold_test = y_tot[startTest:endTest]
    print("%d instances in test set, %d instances in train set" %(len(X_fold_test),len(X_fold_train)))
    c = CorelsClassifier(map_type="prefix", n_iter=10000000, c=0.0001, max_card=1, policy="bfs", bfs_mode=2, useUnfairnessLB=True, fairness=4, maj_pos=UnprotectedIndex+1, min_pos=ProtectedIndex+1, epsilon=0.99, mode=3)
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

kFold = 10 # Enter here the number of folds for the k-fold cross-validation
UnprotectedIndex = 19
ProtectedIndex = 18
X_tot, y_tot, features_tot, prediction_tot = load_from_csv("data/adult_full_binary.csv")
print("--- DATASET INFO --- ")
print("Nombre d'attributs : %d, Nombre d'instances : %d" % (len(features_tot),len(X_tot)))
print("Prediction : %s" %prediction_tot)
print("--------------------")

print("Will perform %d-folds cross-validation" %kFold)
setSize = (len(X_tot)) / kFold
print("Fold size = %d instances" %setSize)


returnList = Parallel(n_jobs=-1)(delayed(performKFold)(foldID=_foldID) for _foldID in [0])#range(kFold))
    

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

#print("-------------------------- Learned rulelist ------------------------------")
#print(c.rl())

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