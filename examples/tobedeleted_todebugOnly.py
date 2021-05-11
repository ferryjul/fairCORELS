import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from collections import Counter
import argparse
from faircorels import load_from_csv, FairCorelsClassifier, ConfusionMatrix, Metric
import csv
import time

N_ITER = 1*10**7 # The maximum number of nodes in the prefix tree
sensitive_attr_column = 0
unsensitive_attr_column = 1

X, y, features, prediction = load_from_csv("./data/adult_rules_full.csv")#("./data/adult_full.csv") # Load the dataset


def computeInconsistentGroups(X, y, verbose=0):
    import pandas as pd
    import numpy as np
    """
    Parameters
    ----------
    X : Features vector
    y : Labels vector
    verbose : Int
    0 -> No display
    1 -> Minimal Display
    1 -> Debug (also performs additional checks)
    Returns
    -------
    Array of tuples: (e_r, k_) for each inconsistent group of examplesriuiuiu
        Where e_r is a representative example of a group of inconsistent examples and
        k_r is the minimum number of instances that can not be classified correctly due to dataset inconsistency, among group r
    """
    representatives = []
    cardinalities = []
    misclassified = []
    values, counts = np.unique(X, axis=0, return_counts=True)
    values = values[counts > 1]
    counts = counts[counts > 1]
    if verbose >= 1:
        print("Found ", values.shape[0], " unique duplicates.")
    incorrCnt = 0
    for ii, anEl in enumerate(list(values)):
        occurences = np.where((X == anEl).all(axis=1))
        representant = occurences[0][0]
        if verbose >= 2:
            print("Value ", anEl, " appears ", counts[ii], " times. (CHECK = ", occurences[0].shape[0], ")")
            print("Occurences: ", occurences, "(representant is instance#", representant, ")")
            # Additional check
            if counts[ii] != occurences[0].shape[0]:
                exit(-1)
        labels = y[occurences[0]]
        if verbose >= 2:
            print(labels)
            # Additional check
            els = X[occurences[0]]
            elsC = np.unique(els, axis=0, return_counts=True)
            if elsC[0].shape[0] > 1:
                exit(-1)
        labelsData = np.unique(labels, return_counts = True)
        if labelsData[0].size > 1:
            if labelsData[0].size != 2: # only two possible values as we work with binary labels -> this case should never happen
                exit(-1)
            minErrors = np.min(labelsData[1])
            if labelsData[1][0] == minErrors: # less 0's
                indicesInLabels = np.where((labels == 0))
                indicesX = occurences[0][indicesInLabels]
                misclassified.extend(indicesX)
                if verbose >= 2:
                    print("Less zeros!")
                    print("associated id label:", indicesInLabels)              
                    print("associated X ids:", indicesX)
            elif labelsData[1][1] == minErrors: # less 1's
                indicesInLabels = np.where((labels == 1))
                indicesX = occurences[0][indicesInLabels]
                misclassified.extend(indicesX)
                if verbose >= 2:
                    print("Less ones!")
                    print("associated id label:", indicesInLabels)     
                    print("associated X ids:", indicesX)
            else:
                print("internal error, exiting")
                exit(-1)
            if verbose >= 2:
                print("min errors possible : ", minErrors)
            incorrCnt += minErrors
            representatives.append(representant)
            cardinalities.append(minErrors)
            #print("Representant = ", representant, ", min errors = ", minErrors)
        else:
            if verbose >= 2:
                print("no inconsistency")
    if verbose >= 1:
        print("At least ", incorrCnt, " elements can not be classified correctly.")
        print("accuracy upper bound = 1 - ", incorrCnt, "/", X.shape[0], " (", 1.0-(incorrCnt/X.shape[0]), ")")        
    return 1.0-(incorrCnt/X.shape[0]), representatives, cardinalities, misclassified

ub, reprs, cards, mis = computeInconsistentGroups(X, y, verbose=0)
#print(mis)

s = 0
for i in range(len(cards)):
    s += cards[i]

print("UB = ", ub)
print("Vals UB = ", 1- (s / X.shape[0]))
print("Misc UB = ", (X.shape[0]-len(mis))/X.shape[0])
