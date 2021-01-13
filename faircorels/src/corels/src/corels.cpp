#include "queue.hh"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "statistical_parity_improved_pruning.cpp"
#include <time.h> // for cp filtering running time measures

Queue::Queue(std::function<bool(Node*, Node*)> cmp, char const *type)
    : q_(new q (cmp)), type_(type) {}

Queue::~Queue() {
    if(q_)
        delete q_;
}

/* Computes confusion matrices for both groups */
bool debug = false; // for printing more info while running/exploring
int pushingTicket = 0;
int pruningCnt = 0;
int permBound = 0;
int arriveHere = 0;
unsigned long exploredNodes = 0;
bool firstPass = true;
bool firstPass2 = true;
unsigned long nodesBeforeBest = 0;
unsigned long cacheBeforeBest = 0;

// Global variables for improved pruning
int nb_sp_plus;
int nb_sp_minus;
int nb_su_plus;
int nb_su_minus;
bool firstCall = true;
int improvedPruningCnt = 0;
int improvedPruningCntTot = 0;
double longestfilteringrun = -1.0;
int best_rl_length = 0;
double total_solver_calls = 0;
long double total_solving_time = 0.0;
solver_args args_longest_run;
int max_depth = 0;
int timeoutCnt = 0;
double max2El(double e1, double e2) {
    if(e1 < e2) {
        return e2;
    } else {
        return e1;
    }
}

double min2El(double e1, double e2) {
    if(e1 > e2) {
        return e2;
    } else {
        return e1;
    }
}

confusion_matrix_groups compute_confusion_matrix(VECTOR parent_prefix_predictions,
                                                CacheTree* tree,
                                                VECTOR parent_not_captured, 
                                                VECTOR captured,  
                                                rule_t* maj_v,
                                                rule_t* min_v,
                                                bool prediction, 
                                                bool default_prediction){

    // datastructures to store the results
    confusion_matrix_groups cmg;
    confusion_matrix cm_minority;
    confusion_matrix cm_majority;

    /*if(firstPass) { // TODO : Update display ? (with group vectors there isn't an attribute name to display ?)
        printf("Fairness calc infos :\n");
        printf("Sensitive attribute : %s, unsensitive attribute : %s\n", tree->rule(min_pos).features, tree->rule(maj_pos).features);
    }*/
    /*if(firstPass)
        printf("Number of rules = %d\n", tree->nrules());*/
    firstPass = false;


    int nsamples = tree->nsamples();
    int pm, num_not_captured;
    VECTOR preds_prefix, not_captured;

    rule_vinit(nsamples, &not_captured);
    rule_vinit(nsamples, &preds_prefix);

    rule_vandnot(not_captured, parent_not_captured, captured, nsamples, &num_not_captured);

    rule_copy(preds_prefix, parent_prefix_predictions, nsamples);
    
    if(default_prediction) { // else it is already OK
        rule_vor(preds_prefix, preds_prefix, not_captured, nsamples, &pm);
    }

    if(prediction) { // else it is already OK
        rule_vor(preds_prefix, preds_prefix, captured, nsamples, &pm);
    }

    // true positives, false negatives, true negatives, and false positives
    VECTOR TP, FP, FN, TN;
    rule_vinit(nsamples, &TP);
    rule_vinit(nsamples, &FP);
    rule_vinit(nsamples, &FN);
    rule_vinit(nsamples, &TN);

    rule_vand(TP, preds_prefix, tree->label(1).truthtable, nsamples, &pm);
    rule_vand(FP, preds_prefix, tree->label(0).truthtable, nsamples, &pm);
    rule_vandnot(FN, tree->label(1).truthtable, preds_prefix, nsamples, &pm);
    rule_vandnot(TN, tree->label(0).truthtable, preds_prefix, nsamples, &pm);

    // true positives, false negatives, true negatives, and false positives for majority group
    VECTOR TP_maj, FP_maj, FN_maj, TN_maj;
    rule_vinit(tree->nsamples(), &TP_maj);
    rule_vinit(tree->nsamples(), &FP_maj);
    rule_vinit(tree->nsamples(), &FN_maj);
    rule_vinit(tree->nsamples(), &TN_maj);

    int nTP_maj, nFP_maj, nFN_maj, nTN_maj;
    rule_vand(TP_maj, TP, maj_v[1].truthtable, nsamples, &nTP_maj);
    rule_vand(FP_maj, FP, maj_v[1].truthtable, nsamples, &nFP_maj);
    rule_vand(FN_maj, FN, maj_v[1].truthtable, nsamples, &nFN_maj);
    rule_vand(TN_maj, TN, maj_v[1].truthtable, nsamples, &nTN_maj);
    
    
    // true positives, false negatives, true negatives, and false positives for minority group
    VECTOR TP_min, FP_min, FN_min, TN_min;
    rule_vinit(nsamples, &TP_min);
    rule_vinit(nsamples, &FP_min);
    rule_vinit(nsamples, &FN_min);
    rule_vinit(nsamples, &TN_min);

    int nTP_min, nFP_min, nFN_min, nTN_min;
    rule_vand(TP_min, TP, min_v[1].truthtable, nsamples, &nTP_min);
    rule_vand(FP_min, FP, min_v[1].truthtable, nsamples, &nFP_min);
    rule_vand(FN_min, FN, min_v[1].truthtable, nsamples, &nFN_min);
    rule_vand(TN_min, TN, min_v[1].truthtable, nsamples, &nTN_min);

    // stats for majority
    double nPPV_maj = (double) nTP_maj / max((nTP_maj + nFP_maj), 1);
    double nTPR_maj = (double) nTP_maj / max((nTP_maj + nFN_maj), 1);

    double nFDR_maj = (double) nFP_maj / max((nFP_maj + nTP_maj), 1);
    double nFPR_maj = (double) nFP_maj / max((nFP_maj + nTN_maj), 1);

    double nFOR_maj = (double) nFN_maj / max((nFN_maj + nTN_maj), 1);
    double nFNR_maj = (double) nFN_maj / max((nFN_maj + nTP_maj), 1);

    double nNPV_maj = (double) nTN_maj / max((nTN_maj + nFN_maj), 1);
    double nTNR_maj = (double) nTN_maj / max((nTN_maj + nFP_maj), 1);

    cm_majority.nTP = nTP_maj;
    cm_majority.nFP = nFP_maj;
    cm_majority.nFN = nFN_maj;
    cm_majority.nTN = nTN_maj;

    cm_majority.nPPV = nPPV_maj;
    cm_majority.nTPR = nTPR_maj;
    cm_majority.nFDR = nFDR_maj;
    cm_majority.nFPR = nFPR_maj;
    cm_majority.nFOR = nFOR_maj;
    cm_majority.nFNR = nFNR_maj;
    cm_majority.nNPV = nNPV_maj;
    cm_majority.nTNR = nTNR_maj;


    // stats for minority
    double nPPV_min = (double) nTP_min / max((nTP_min + nFP_min), 1);
    double nTPR_min = (double) nTP_min / max((nTP_min + nFN_min), 1);
    double nFDR_min = (double) nFP_min / max((nFP_min + nTP_min), 1);
    double nFPR_min = (double) nFP_min / max((nFP_min + nTN_min), 1);
    double nFOR_min = (double) nFN_min / max((nFN_min + nTN_min), 1);
    double nFNR_min = (double) nFN_min / max((nFN_min + nTP_min), 1);
    double nNPV_min = (double) nTN_min / max((nTN_min + nFN_min), 1);
    double nTNR_min = (double) nTN_min / max((nTN_min + nFP_min), 1);

    cm_minority.nTP = nTP_min;
    cm_minority.nFP = nFP_min;
    cm_minority.nFN = nFN_min;
    cm_minority.nTN = nTN_min;

    cm_minority.nPPV = nPPV_min;
    cm_minority.nTPR = nTPR_min;
    cm_minority.nFDR = nFDR_min;
    cm_minority.nFPR = nFPR_min;
    cm_minority.nFOR = nFOR_min;
    cm_minority.nFNR = nFNR_min;
    cm_minority.nNPV = nNPV_min;
    cm_minority.nTNR = nTNR_min;

    cmg.majority = cm_majority;
    cmg.minority = cm_minority;

    /* UPPER BOUND CALC */
    // true positives, false negatives, true negatives, and false positives tables for minority group
    VECTOR TP_min_ub, FN_min_ub, TN_min_ub, FP_min_ub;
    rule_vinit(tree->nsamples(), &TP_min_ub);
    rule_vinit(tree->nsamples(), &FN_min_ub);
    rule_vinit(tree->nsamples(), &TN_min_ub);
    rule_vinit(tree->nsamples(), &FP_min_ub);
    VECTOR TP_maj_ub, FN_maj_ub, TN_maj_ub, FP_maj_ub;
    rule_vinit(tree->nsamples(), &TP_maj_ub);
    rule_vinit(tree->nsamples(), &FN_maj_ub);
    rule_vinit(tree->nsamples(), &TN_maj_ub);
    rule_vinit(tree->nsamples(), &FP_maj_ub);

    int nTP_maj_ub;
    int nFN_maj_ub;
    int nTN_maj_ub;
    int nFP_maj_ub;
    // for lower bounds we do not consider instances captured by the default decision
    rule_vandnot(TP_maj_ub, TP_maj, not_captured, tree->nsamples(), &nTP_maj_ub);
    rule_vandnot(FN_maj_ub, FN_maj, not_captured, tree->nsamples(), &nFN_maj_ub);
    rule_vandnot(FP_maj_ub, FP_maj, not_captured, tree->nsamples(), &nFP_maj_ub);
    rule_vandnot(TN_maj_ub, TN_maj, not_captured, tree->nsamples(), &nTN_maj_ub);

    int nTP_min_ub;
    int nFN_min_ub;
    int nTN_min_ub;
    int nFP_min_ub;

    rule_vandnot(TP_min_ub, TP_min, not_captured, tree->nsamples(), &nTP_min_ub);
    rule_vandnot(FN_min_ub, FN_min, not_captured, tree->nsamples(), &nFN_min_ub);
    rule_vandnot(FP_min_ub, FP_min, not_captured, tree->nsamples(), &nFP_min_ub);
    rule_vandnot(TN_min_ub, TN_min, not_captured, tree->nsamples(), &nTN_min_ub);

    int totMAJ = nTP_maj + nFN_maj + nFP_maj + nTN_maj;
    int totMIN = nTP_min + nFN_min + nFP_min + nTN_min;

    double B1 = (double)((double)(totMAJ - nFN_maj_ub - nTN_maj_ub)/(double)totMAJ);
    double B2 = (double) ((double)(nTP_maj_ub + nFP_maj_ub)/(double)totMAJ);
    double B3 = (double) ((double)(totMIN - nFN_min_ub - nTN_min_ub)/(double)totMIN);
    double B4 = (double) ((double)(nTP_min_ub + nFP_min_ub)/(double)totMIN);
    double min_val = 0;
    if(B1 < B2 || B3 < B4) {
        printf("problem !\n");
        exit(-1);
    }
    if(B3 < B2) {
        min_val = (B2-B3);
    } else if(B4 > B1) {
        min_val = (B4-B1);
    } else {
        min_val = 0;
    }

    cmg.unfairnessLB = min_val;

    /* OTHER METRICS */
    
    int totPos_maj = nTP_maj + nFN_maj;
    int totPos_min = nTP_min + nFN_min;
    int totNeg_maj = nTN_maj + nFP_maj;
    int totNeg_min = nTN_min + nFP_min;

    int maxFP_maj = totNeg_maj - nTN_maj_ub;
    int maxFP_min = totNeg_min - nTN_min_ub;
    int minFP_maj = nFP_maj_ub;
    int minFP_min = nFP_min_ub;

    int maxTP_maj = totPos_maj - nFN_maj_ub;
    int maxTP_min = totPos_min - nFN_min_ub;
    int minTP_maj = nTP_maj_ub;
    int minTP_min = nTP_min_ub;

    int maxTN_maj = totNeg_maj - nFP_maj_ub;
    int maxTN_min = totNeg_min - nFP_min_ub;
    int minTN_maj = nTN_maj_ub;
    int minTN_min = nTN_min_ub;

    int maxFN_maj = totPos_maj - nTP_maj_ub;
    int maxFN_min = totPos_min - nTP_min_ub;
    int minFN_maj = nFN_maj_ub;
    int minFN_min = nFN_min_ub;

    /* Predictive parity */
    // Bound (A)
    double PPV_maj_min1 = (double) (minTP_maj) / (double) (maxTP_maj + maxFP_maj);
    double PPV_maj_max1 = (double) (maxTP_maj) / (double) (minTP_maj + minFP_maj);
    double PPV_min_min1 = (double) (minTP_min) / (double) (maxTP_min + maxFP_min);
    double PPV_min_max1 = (double) (maxTP_min) / (double) (minTP_min + minFP_min);
    // Bound (B)
    double PPV_maj_min2 = ((double) (minTP_maj + minFP_maj - maxFP_maj))/((double) (minTP_maj + minFP_maj));
    double PPV_maj_max2 = ((double) (maxTP_maj + maxFP_maj - minFP_maj))/((double)(maxTP_maj + maxFP_maj));
    double PPV_min_min2 = ((double) (minTP_min + minFP_min - maxFP_min))/((double) (minTP_min + minFP_min));
    double PPV_min_max2 = ((double) (maxTP_min + maxFP_min - minFP_min))/((double)(maxTP_min + maxFP_min));
    // Take the best bound
    double PPV_maj_min = max2El(PPV_maj_min1, PPV_maj_min2);
    double PPV_maj_max = min2El(PPV_maj_max1, PPV_maj_max2);
    double PPV_min_min = max2El(PPV_min_min1, PPV_min_min2);
    double PPV_min_max = min2El(PPV_min_max1, PPV_min_max2);
    // Corrects max values in case they are too big (max value might be very bad)
    if(PPV_maj_max > 1)
        PPV_maj_max = 1;
    if(PPV_min_max > 1)
        PPV_min_max = 1;

    // Compute distance between intervals
    double min_pred_par = 0;
    if(PPV_maj_min > PPV_min_max) {
        min_pred_par = PPV_maj_min - PPV_min_max;
    }
    if(PPV_min_min > PPV_maj_max) {
        min_pred_par = PPV_min_min - PPV_maj_max;
    }
    cmg.predparityLB = min_pred_par;

    /* Predictive equality */
    // Bound (A)
    double FPR_maj_min1 = (double) (minFP_maj) / (double) min2El((maxTN_maj + maxFP_maj),totNeg_maj);
    double FPR_maj_max1 = (double) (maxFP_maj) / (double) (minTN_maj + minFP_maj);
    double FPR_min_min1 = (double) (minFP_min) / (double) min2El((maxTN_min + maxFP_min),totNeg_min);
    double FPR_min_max1 = (double) (maxFP_min) / (double) (minTN_min + minFP_min);
    // Bound (B)
    double FPR_maj_min2 = ((double) (max2El((minFP_maj + minTN_maj),totNeg_maj) - maxTN_maj))/((double) max2El((minFP_maj + minTN_maj),totNeg_maj));
    double FPR_maj_max2 = ((double) (min2El((maxTN_maj + maxFP_maj),totNeg_maj) - minTN_maj))/((double) min2El((maxTN_maj + maxFP_maj),totNeg_maj));
    double FPR_min_min2 = ((double) (max2El((minFP_min + minTN_min),totNeg_min) - maxTN_min))/((double) max2El((minFP_min + minTN_min),totNeg_min));
    double FPR_min_max2 = ((double) (min2El((maxTN_min + maxFP_min),totNeg_min) - minTN_min))/((double) min2El((maxTN_min + maxFP_min),totNeg_min));
    // Take the best bound
    double FPR_maj_min = max2El(FPR_maj_min1, FPR_maj_min2);
    double FPR_maj_max = min2El(FPR_maj_max1, FPR_maj_max2);
    double FPR_min_min = max2El(FPR_min_min1, FPR_min_min2);
    double FPR_min_max = min2El(FPR_min_max1, FPR_min_max2);
    // Corrects max values in case they are too big (max value might be very bad)
    if(FPR_maj_max > 1)
        FPR_maj_max = 1;
    if(FPR_min_max > 1)
        FPR_min_max = 1;
    
    // Compute distance between intervals
    double min_pred_equ = 0;
    if(FPR_maj_min > FPR_min_max) {
        min_pred_equ = FPR_maj_min - FPR_min_max;
    }
    if(FPR_min_min > FPR_maj_max) {
        min_pred_equ = FPR_min_min - FPR_maj_max;
    }
    cmg.predequalityLB = min_pred_equ;

    /* Equal opportunity */
    // Bound (A)
    double FNR_maj_min1 = (double) (minFN_maj) / (double) min2El((maxTP_maj + maxFN_maj), totPos_maj);
    double FNR_maj_max1 = (double) (maxFN_maj) / (double) (minTP_maj + minFN_maj);
    double FNR_min_min1 = (double) (minFN_min) / (double) min2El((maxTP_min + maxFN_min), totPos_min);
    double FNR_min_max1 = (double) (maxFN_min) / (double) (minTP_min + minFN_min);
    // Bound (B)
    double FNR_maj_min2 = ((double) (max2El((minFN_maj + minTP_maj),totPos_maj) - maxTP_maj))/((double) max2El((minFN_maj + minTP_maj),totPos_maj));
    double FNR_maj_max2 = ((double) (min2El((maxTP_maj + maxFN_maj), totPos_maj) - minTP_maj))/((double)(min2El((maxTP_maj + maxFN_maj), totPos_maj)));
    double FNR_min_min2 = ((double) (max2El((minFN_min + minTP_min),totPos_min) - maxTP_min))/((double) max2El((minFN_min + minTP_min),totPos_min));
    double FNR_min_max2 = ((double) (min2El((maxTP_min + maxFN_min), totPos_min) - minTP_min))/((double)(min2El((maxTP_min + maxFN_min), totPos_min)));
    // Take the best bound
    double FNR_maj_min = max2El(FNR_maj_min1, FNR_maj_min2);
    double FNR_maj_max = min2El(FNR_maj_max1, FNR_maj_max2);
    double FNR_min_min = max2El(FNR_min_min1, FNR_min_min2);
    double FNR_min_max = min2El(FNR_min_max1, FNR_min_max2);
    // Corrects max values in case they are too big (max value might be very bad)
    if(FNR_maj_max > 1)
        FNR_maj_max = 1;
    if(FNR_min_max > 1)
        FNR_min_max = 1;

    // Compute distance between intervals
    double min_equ_op = 0;

    if(FNR_maj_min > FNR_min_max) {
        min_equ_op = FNR_maj_min - FNR_min_max;
    }
    if(FNR_min_min > FNR_maj_max) {
        min_equ_op = FNR_min_min - FNR_maj_max;
    }
    cmg.equalOppLB = min_equ_op;

    /* ADDITIONNAL INFO, FOR DEBUGGING */
    cmg.minority.min_ppv = PPV_min_min;
    cmg.minority.max_ppv = PPV_min_max;
    cmg.majority.min_ppv = PPV_maj_min;
    cmg.majority.max_ppv = PPV_maj_max;
    cmg.minority.min_fpr = FPR_min_min;
    cmg.minority.max_fpr = FPR_min_max;
    cmg.majority.min_fpr = FPR_maj_min;
    cmg.majority.max_fpr = FPR_maj_max;
    cmg.minority.min_fnr = FNR_min_min;
    cmg.minority.max_fnr = FNR_min_max;
    cmg.majority.min_fnr = FNR_maj_min;
    cmg.majority.max_fnr = FNR_maj_max;
    cmg.minority.min_tp = minTP_min;
    cmg.majority.min_tp = minTP_maj;
    cmg.minority.max_tp = maxTP_min;
    cmg.majority.max_tp = maxTP_maj;
    cmg.minority.min_fp = minFP_min;
    cmg.majority.min_fp = minFP_maj;
    cmg.minority.max_fp = maxFP_min;
    cmg.majority.max_fp = maxFP_maj;
    cmg.minority.min_tn = minTN_min;
    cmg.majority.min_tn = minTN_maj;
    cmg.minority.max_tn = maxTN_min;
    cmg.majority.max_tn = maxTN_maj;
    cmg.minority.min_fn = minFN_min;
    cmg.majority.min_fn = minFN_maj;
    cmg.minority.max_fn = maxFN_min;
    cmg.majority.max_fn = maxFN_maj;
    rule_vfree(&not_captured);
    rule_vfree(&preds_prefix);
    rule_vfree(&TP);
    rule_vfree(&FP);
    rule_vfree(&FN);
    rule_vfree(&TN);
    rule_vfree(&TP_maj);
    rule_vfree(&FP_maj);
    rule_vfree(&FN_maj);
    rule_vfree(&TN_maj);
    rule_vfree(&TP_min);
    rule_vfree(&FP_min);
    rule_vfree(&FN_min);
    rule_vfree(&TN_min);
    rule_vfree(&TP_maj_ub);
    rule_vfree(&FN_maj_ub);
    rule_vfree(&TN_maj_ub);
    rule_vfree(&FP_maj_ub);
    rule_vfree(&TP_min_ub);
    rule_vfree(&FN_min_ub);
    rule_vfree(&TN_min_ub);
    rule_vfree(&FP_min_ub);

    return cmg;
}

/* Computes fairness metrics given confusion matrices of both groups */
fairness_metrics compute_fairness_metrics(confusion_matrix_groups cmg){
    fairness_metrics metrics;
    
    // statistical_parity
    double statistical_parity_maj = (double) (cmg.majority.nTP + cmg.majority.nFP) / 
                                                max((cmg.majority.nTP + cmg.majority.nFP + cmg.majority.nFN + cmg.majority.nTN),1);
                                
    double statistical_parity_min = (double) (cmg.minority.nTP + cmg.minority.nFP) / 
                                                max((cmg.minority.nTP + cmg.minority.nFP + cmg.minority.nFN + cmg.minority.nTN),1);
                                
                                
    metrics.statistical_parity =  fabs(statistical_parity_maj - statistical_parity_min);

    // predictive parity
    metrics.predictive_parity = fabs(cmg.majority.nPPV - cmg.minority.nPPV);

    // predictive equality
    metrics.predictive_equality = fabs(cmg.majority.nFPR - cmg.minority.nFPR);

    // equal opportunity
    metrics.equal_opportunity = fabs(cmg.majority.nFNR - cmg.minority.nFNR);

    // equalized_odds
    metrics.equalized_odds = fabs(cmg.majority.nFNR - cmg.minority.nFNR) + fabs(cmg.majority.nFPR - cmg.minority.nFPR);

    // cond_use_acc_equality
    metrics.cond_use_acc_equality = fabs(cmg.majority.nPPV - cmg.minority.nPPV) + fabs(cmg.majority.nNPV - cmg.minority.nNPV);

    return metrics;
}


/*
 * Performs incremental computation on a node, evaluating the bounds and inserting into the cache,
 * queue, and permutation map if appropriate.
 * This is the function that contains the majority of the logic of the algorithm.
 *
 * parent -- the node that is going to have all of its children evaluated.
 * parent_not_captured -- the vector representing data points NOT captured by the parent.
 */
void evaluate_children(CacheTree* tree, 
                        Node* parent, 
                        tracking_vector<unsigned short, DataStruct::Tree> parent_prefix,
                        VECTOR parent_not_captured, 
                        Queue* q, 
                        PermutationMap* p,
                        double beta,
                        int fairness,
                        rule_t* maj_v,
                        rule_t* min_v,
                        int mode,
                        bool useUnfairnessLB,
                        double min_fairness_acceptable,
                        bool forbidSensAttr,
                        double accuracyUpperBound){
    
    if(firstCall){
        firstCall = false;
        VECTOR captured_it;
        rule_vinit(tree->nsamples(), &captured_it);
        rule_vand(captured_it, min_v[1].truthtable, tree->label(1).truthtable, tree->nsamples(), &nb_sp_plus);
        rule_vand(captured_it, min_v[1].truthtable, tree->label(0).truthtable, tree->nsamples(), &nb_sp_minus);
        rule_vand(captured_it, maj_v[1].truthtable, tree->label(1).truthtable, tree->nsamples(), &nb_su_plus);
        rule_vand(captured_it, maj_v[1].truthtable, tree->label(0).truthtable, tree->nsamples(), &nb_su_minus);
        if(debug) {
            printf("Initializing cardinalities for SP improved pruning : \n");
            printf("Got %d protected positives, %d protected negatives, %d unprotected positives, %d unprotected negatives.\n", nb_sp_plus, nb_sp_minus, nb_su_plus, nb_su_minus);
        }
        rule_vfree(&captured_it);
        int U = accuracyUpperBound * (tree->nsamples());
        if(debug) {
            printf("U is %d/%d.\n", U, tree->nsamples());
        }
        if(debug) {
            if(fairness == 1 && useUnfairnessLB)
                printf("will perform improved SP pruning\n");
            else if(fairness == 2 && useUnfairnessLB)
                printf("will perform improved PP pruning\n");
            else if(fairness == 3 && useUnfairnessLB)
                printf("will perform improved PE pruning\n");
            else if(fairness == 4 && useUnfairnessLB)
                printf("will perform improved EO pruning\n");
            else if(fairness == 5 && useUnfairnessLB)
                printf("will perform improved EOdds pruning\n");
        }
        longestfilteringrun = -1;
        nodesBeforeBest = 0;
        cacheBeforeBest = 0;      
    }
    VECTOR captured, captured_zeros, not_captured, not_captured_zeros, not_captured_equivalent;
    int num_captured, c0, c1, captured_correct;
    int num_not_captured, d0, d1, default_correct, num_not_captured_equivalent;
    bool prediction, default_prediction;
    double lower_bound, objective, parent_lower_bound, lookahead_bound;
    double parent_equivalent_minority;
    double equivalent_minority = 0.;
    int nsamples = tree->nsamples();
    int nrules = tree->nrules();
    double c = tree->c();
    double threshold = c * nsamples;
    rule_vinit(nsamples, &captured);
    rule_vinit(nsamples, &captured_zeros);
    rule_vinit(nsamples, &not_captured);
    rule_vinit(nsamples, &not_captured_zeros);
    rule_vinit(nsamples, &not_captured_equivalent);
    int i, len_prefix;
    len_prefix = parent->depth() + 1;
    parent_lower_bound = parent->lower_bound();
    parent_equivalent_minority = parent->equivalent_minority();
    //int verbosity = logger->getVerbosity();
    double t0 = timestamp();

    // Compute prefix's predictions
    VECTOR captured_it, not_captured_yet, captured_zeros_j, preds_prefix;

    int nb, nb2, pm;

    rule_vinit(tree->nsamples(), &captured_it);
    rule_vinit(tree->nsamples(), &not_captured_yet);
    rule_vinit(tree->nsamples(), &preds_prefix);
    rule_vinit(tree->nsamples(), &captured_zeros_j);

    // Initially not_captured_yet is full of ones
    rule_vor(not_captured_yet, tree->label(0).truthtable, tree->label(1).truthtable, tree->nsamples(),&nb);

    // Initially preds_prefix is full of zeros
    rule_vclear(tree->nsamples(), preds_prefix);
    int depth = len_prefix;
    tracking_vector<unsigned short, DataStruct::Tree>::iterator it;
    /*--- tobedeleted ---*/
    /*bool isPrefixCorresp = true;
    if (len_prefix == 4)  {
        int ind = 0;
        for (it = parent_prefix.begin(); it != parent_prefix.end() && isPrefixCorresp; it++) {
            switch(ind){
                case 0: 
                    if(strcmp(tree->rule(*it).features,"age_26-45__AND__juvenile-crimes_=0"))
                        isPrefixCorresp = false;
                    break;
                case 1: 
                    if(strcmp(tree->rule(*it).features,"gender_Male__AND__juvenile-crimes_>0"))
                        isPrefixCorresp = false;
                    break;
                case 2: 
                    if(strcmp(tree->rule(*it).features,"priors_2-3"))
                        isPrefixCorresp = false;
                    break;
                
            }
            ind++;
        }   
    } else {
        isPrefixCorresp = false;
    }
    if(isPrefixCorresp){
        printf("Prefix found !\n");
    }*/
    /* --- --- */
    for (it = parent_prefix.begin(); it != parent_prefix.end(); it++) {
        rule_vand(captured_it, not_captured_yet, tree->rule(*it).truthtable, tree->nsamples(), &nb);
        rule_vandnot(not_captured_yet, not_captured_yet, captured_it, tree->nsamples(), &pm);
        rule_vand(captured_zeros_j, captured_it, tree->label(0).truthtable, tree->nsamples(), &nb2);
        if(nb2 <= (nb - nb2)) { //then prediction is 1
            rule_vor(preds_prefix, preds_prefix, captured_it, tree->nsamples(), &nb);
        }
    }
    
    // begin evaluating children
    bool pass;
    for (i = 1; i < nrules; i++) {
        pass = false;
        /*if (isPrefixCorresp && !strcmp(tree-> rule(i).features,"priors_>3__AND__age_23-25")){
            printf("rule list evaluated ! \n");
        }*/
        /* IF RULE CORRESPONDS TO PROTECTED OR UNPROTECTED ATTRIBUTES, IT IS PRUNED */
        /*if(forbidSensAttr) { // TODO : RE IMPLEMENT THIS PART WITH GROUP VECTORS (?) - OR DELETE IF REPLACED BY RULE MINING
            if(i == maj_pos || i == min_pos) {
                if(firstPass2) {
                    //printf("pruning subtree with rules %s or %s\n", tree->rule(maj_pos).features, tree->rule(min_pos).features);
                    firstPass2 = false;
                }
                continue;
            }
        }*/
        //printf("rule : %d/%d, %s\n", i, nrules, tree->rule(i).features);
        double t1 = timestamp();
        // check if this rule is already in the prefix
        if (std::find(parent_prefix.begin(), parent_prefix.end(), i) != parent_prefix.end())
            continue;
        exploredNodes++; // we consider node explored here as we below compute the preds & captured instances
        // captured represents data captured by the new rule
        rule_vand(captured, parent_not_captured, tree->rule(i).truthtable, nsamples, &num_captured);
        // lower bound on antecedent support
        if ((tree->ablation() != 1 && tree->ablation() != 3) && (num_captured < threshold))
            continue;
        rule_vand(captured_zeros, captured, tree->label(0).truthtable, nsamples, &c0);
        c1 = num_captured - c0;
        if (c0 > c1) {
            prediction = 0;
            captured_correct = c0;
        } else {
            prediction = 1;
            captured_correct = c1;
        }
        // lower bound on accurate antecedent support
        if ((tree->ablation() != 1 && tree->ablation() != 3) && (captured_correct < threshold))
            continue;
        // subtract off parent equivalent points bound because we want to use pure lower bound from parent
        lower_bound = parent_lower_bound - parent_equivalent_minority + (double)(num_captured - captured_correct) / nsamples + c;
        logger->addToLowerBoundTime(time_diff(t1));
        logger->incLowerBoundNum();
        if (lower_bound >= tree->min_objective()) // hierarchical objective lower bound
	        continue;
        double t2 = timestamp();
        rule_vandnot(not_captured, parent_not_captured, captured, nsamples, &num_not_captured);
        rule_vand(not_captured_zeros, not_captured, tree->label(0).truthtable, nsamples, &d0);
        d1 = num_not_captured - d0;
        if (d0 > d1) {
            default_prediction = 0;
            default_correct = d0;
        } else {
            default_prediction = 1;
            default_correct = d1;
        }

        //double misc = (double)(num_not_captured - default_correct) / nsamples;
        double unfairness = 0.0;

        confusion_matrix_groups cmg = compute_confusion_matrix(preds_prefix, tree, parent_not_captured, captured,
                                                                                 maj_v, min_v, prediction, default_prediction);

        improvedPruningCntTot++;
        // for statistical parity, we use constraint programming to improve pruning     
        if(depth > max_depth && debug){
            max_depth = depth;
            printf("Now working at depth %d.\n", max_depth);
        }    

        // ----------------------------------- HERE OCCURS THE PPC FILTERING -----------------------------------                                      
        if(fairness == 1 && useUnfairnessLB && best_rl_length > 0){  // Statistical Parity
			int L = (1 - (tree->min_objective() + ((best_rl_length-len_prefix)*c)))*tree->nsamples(); // (1 - misc)*nb_samples = nb inst well classif by current best model
            //if(improvedPruningCntTot % 10000 == 0)
            //    printf("new lower bound : %f\n", (tree->min_objective() + ((best_rl_length-len_prefix)*c)));
			int U = accuracyUpperBound * (tree->nsamples());
			float fairness_tolerence = 1-min_fairness_acceptable; // equiv max unfairness acceptable
			int TPp = cmg.minority.min_tp;
			int FPp = cmg.minority.min_fp;
			int TNp = cmg.minority.min_tn;
			int FNp = cmg.minority.min_fn;
			int TPu = cmg.majority.min_tp;
			int FPu = cmg.majority.min_fp;
			int TNu = cmg.majority.min_tn;
			int FNu = cmg.majority.min_fn;
                       
            // Start measuring time
            struct timespec begin, end; 
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &begin); //CLOCK_REALTIME for wall clock time, CLOCK_PROCESS_CPUTIME_ID for CPU time

            FilteringStatisticalParity check_bounds(nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , fairness_tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
            check_bounds.run(0);

            // Stop measuring time and calculate the elapsed time
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
            total_solver_calls+=1;
            long seconds = end.tv_sec - begin.tv_sec;
            long nanoseconds = end.tv_nsec - begin.tv_nsec;
            double timediff = (seconds*1000000) + (nanoseconds*1e-3);
            total_solving_time += timediff;
            if(timediff > longestfilteringrun){
                longestfilteringrun = timediff;
                args_longest_run.nb_sp_plus = nb_sp_plus;
                args_longest_run.nb_sp_minus = nb_sp_minus;
                args_longest_run.nb_su_plus = nb_su_plus;
                args_longest_run.nb_su_minus = nb_su_minus;
                args_longest_run.L = L;
                args_longest_run.U = U;
                args_longest_run.fairness_tolerence = fairness_tolerence;
                args_longest_run.TPp = TPp;
                args_longest_run.FPp = FPp;
                args_longest_run.TNp = TNp;
                args_longest_run.FNp = FNp;
                args_longest_run.TPu = TPu;
                args_longest_run.FPu = FPu;
                args_longest_run.TNu = TNu;
                args_longest_run.FNu = FNu;
            }
            if(!check_bounds.isFeasible()){ // no solution => the fairness constraint can never be satisfied using the current prefix -> we skip its evaluation without adding it to the queue
                improvedPruningCnt++;
                continue;
            }   
        } else if(fairness == 2 && useUnfairnessLB && best_rl_length > 0){  // Predictive Parity
			int L = (1 - (tree->min_objective() + ((best_rl_length-len_prefix)*c)))*tree->nsamples(); // (1 - misc)*nb_samples = nb inst well classif by current best model
			int U = accuracyUpperBound * (tree->nsamples());
			float fairness_tolerence = 1-min_fairness_acceptable; // equiv max unfairness acceptable
			int TPp = cmg.minority.min_tp;
			int FPp = cmg.minority.min_fp;
			int TNp = cmg.minority.min_tn;
			int FNp = cmg.minority.min_fn;
			int TPu = cmg.majority.min_tp;
			int FPu = cmg.majority.min_fp;
			int TNu = cmg.majority.min_tn;
			int FNu = cmg.majority.min_fn;
                       
            // Start measuring time
            struct timespec begin, end; 
            //printf("Calling solver with parameters : (%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d,%d,%d,%d,%d)\n", nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , fairness_tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &begin); //CLOCK_REALTIME for wall clock time, CLOCK_PROCESS_CPUTIME_ID for CPU time
            FilteringPredictiveParity check_bounds(nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , fairness_tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
            check_bounds.run(0);

            // Stop measuring time and calculate the elapsed time
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
            total_solver_calls+=1;
            long seconds = end.tv_sec - begin.tv_sec;
            long nanoseconds = end.tv_nsec - begin.tv_nsec;
            double timediff = (seconds*1000000) + (nanoseconds*1e-3);
            total_solving_time += timediff;
            if(timediff > longestfilteringrun){
                longestfilteringrun = timediff;
                args_longest_run.nb_sp_plus = nb_sp_plus;
                args_longest_run.nb_sp_minus = nb_sp_minus;
                args_longest_run.nb_su_plus = nb_su_plus;
                args_longest_run.nb_su_minus = nb_su_minus;
                args_longest_run.L = L;
                args_longest_run.U = U;
                args_longest_run.fairness_tolerence = fairness_tolerence;
                args_longest_run.TPp = TPp;
                args_longest_run.FPp = FPp;
                args_longest_run.TNp = TNp;
                args_longest_run.FNp = FNp;
                args_longest_run.TPu = TPu;
                args_longest_run.FPu = FPu;
                args_longest_run.TNu = TNu;
                args_longest_run.FNu = FNu;
            }
            if(!check_bounds.isFeasible()){ // no solution => the fairness constraint can never be satisfied using the current prefix -> we skip its evaluation without adding it to the queue
                improvedPruningCnt++;
                continue;
            }   
        } else if(fairness == 3 && useUnfairnessLB && best_rl_length > 0){  // Predictive Parity
			int L = (1 - (tree->min_objective() + ((best_rl_length-len_prefix)*c)))*tree->nsamples(); // (1 - misc)*nb_samples = nb inst well classif by current best model
			int U = accuracyUpperBound * (tree->nsamples());
			float fairness_tolerence = 1-min_fairness_acceptable; // equiv max unfairness acceptable
			int TPp = cmg.minority.min_tp;
			int FPp = cmg.minority.min_fp;
			int TNp = cmg.minority.min_tn;
			int FNp = cmg.minority.min_fn;
			int TPu = cmg.majority.min_tp;
			int FPu = cmg.majority.min_fp;
			int TNu = cmg.majority.min_tn;
			int FNu = cmg.majority.min_fn;
                       
            // Start measuring time
            struct timespec begin, end; 
            //printf("Calling solver with parameters : (%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d,%d,%d,%d,%d)\n", nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , fairness_tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &begin); //CLOCK_REALTIME for wall clock time, CLOCK_PROCESS_CPUTIME_ID for CPU time
            FilteringPredictiveEquality check_bounds(nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , fairness_tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
            check_bounds.run(0);

            // Stop measuring time and calculate the elapsed time
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
            total_solver_calls+=1;
            long seconds = end.tv_sec - begin.tv_sec;
            long nanoseconds = end.tv_nsec - begin.tv_nsec;
            double timediff = (seconds*1000000) + (nanoseconds*1e-3);
            total_solving_time += timediff;
            if(timediff > longestfilteringrun){
                longestfilteringrun = timediff;
                args_longest_run.nb_sp_plus = nb_sp_plus;
                args_longest_run.nb_sp_minus = nb_sp_minus;
                args_longest_run.nb_su_plus = nb_su_plus;
                args_longest_run.nb_su_minus = nb_su_minus;
                args_longest_run.L = L;
                args_longest_run.U = U;
                args_longest_run.fairness_tolerence = fairness_tolerence;
                args_longest_run.TPp = TPp;
                args_longest_run.FPp = FPp;
                args_longest_run.TNp = TNp;
                args_longest_run.FNp = FNp;
                args_longest_run.TPu = TPu;
                args_longest_run.FPu = FPu;
                args_longest_run.TNu = TNu;
                args_longest_run.FNu = FNu;
            }
            if(!check_bounds.isFeasible()){ // no solution => the fairness constraint can never be satisfied using the current prefix -> we skip its evaluation without adding it to the queue
                improvedPruningCnt++;
                pass = true;
            }   
        } else if(fairness == 4 && useUnfairnessLB && best_rl_length > 0){  // Predictive Parity
			int L = (1 - (tree->min_objective() + ((best_rl_length-len_prefix)*c)))*tree->nsamples(); // (1 - misc)*nb_samples = nb inst well classif by current best model
			int U = accuracyUpperBound * (tree->nsamples());
			float fairness_tolerence = 1-min_fairness_acceptable; // equiv max unfairness acceptable
			int TPp = cmg.minority.min_tp;
			int FPp = cmg.minority.min_fp;
			int TNp = cmg.minority.min_tn;
			int FNp = cmg.minority.min_fn;
			int TPu = cmg.majority.min_tp;
			int FPu = cmg.majority.min_fp;
			int TNu = cmg.majority.min_tn;
			int FNu = cmg.majority.min_fn;
                       
            // Start measuring time
            struct timespec begin, end; 
            //printf("Calling solver with parameters : (%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d,%d,%d,%d,%d)\n", nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , fairness_tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &begin); //CLOCK_REALTIME for wall clock time, CLOCK_PROCESS_CPUTIME_ID for CPU time
            FilteringEqualOpportunity check_bounds(nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , fairness_tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
            check_bounds.run(0);

            // Stop measuring time and calculate the elapsed time
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
            total_solver_calls+=1;
            long seconds = end.tv_sec - begin.tv_sec;
            long nanoseconds = end.tv_nsec - begin.tv_nsec;
            double timediff = (seconds*1000000) + (nanoseconds*1e-3);
            total_solving_time += timediff;
            if(timediff > longestfilteringrun){
                longestfilteringrun = timediff;
                args_longest_run.nb_sp_plus = nb_sp_plus;
                args_longest_run.nb_sp_minus = nb_sp_minus;
                args_longest_run.nb_su_plus = nb_su_plus;
                args_longest_run.nb_su_minus = nb_su_minus;
                args_longest_run.L = L;
                args_longest_run.U = U;
                args_longest_run.fairness_tolerence = fairness_tolerence;
                args_longest_run.TPp = TPp;
                args_longest_run.FPp = FPp;
                args_longest_run.TNp = TNp;
                args_longest_run.FNp = FNp;
                args_longest_run.TPu = TPu;
                args_longest_run.FPu = FPu;
                args_longest_run.TNu = TNu;
                args_longest_run.FNu = FNu;
            }
            if(!check_bounds.isFeasible()){ // no solution => the fairness constraint can never be satisfied using the current prefix -> we skip its evaluation without adding it to the queue
                improvedPruningCnt++;
                continue;
            }   
        } else if(fairness == 5 && useUnfairnessLB && best_rl_length > 0){  // Predictive Parity
			int L = (1 - (tree->min_objective() + ((best_rl_length-len_prefix)*c)))*tree->nsamples(); // (1 - misc)*nb_samples = nb inst well classif by current best model
			int U = accuracyUpperBound * (tree->nsamples());
			float fairness_tolerence = 1-min_fairness_acceptable; // equiv max unfairness acceptable
			int TPp = cmg.minority.min_tp;
			int FPp = cmg.minority.min_fp;
			int TNp = cmg.minority.min_tn;
			int FNp = cmg.minority.min_fn;
			int TPu = cmg.majority.min_tp;
			int FPu = cmg.majority.min_fp;
			int TNu = cmg.majority.min_tn;
			int FNu = cmg.majority.min_fn;
                       
            // Start measuring time
            struct timespec begin, end; 
            //printf("Calling solver with parameters : (%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d,%d,%d,%d,%d)\n", nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , fairness_tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &begin); //CLOCK_REALTIME for wall clock time, CLOCK_PROCESS_CPUTIME_ID for CPU time
            FilteringEqualizedOdds check_bounds(nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , fairness_tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
            check_bounds.run(0);

            // Stop measuring time and calculate the elapsed time
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
            total_solver_calls+=1;
            long seconds = end.tv_sec - begin.tv_sec;
            long nanoseconds = end.tv_nsec - begin.tv_nsec;
            double timediff = (seconds*1000000) + (nanoseconds*1e-3);
            total_solving_time += timediff;
            if(timediff > longestfilteringrun){
                longestfilteringrun = timediff;
                args_longest_run.nb_sp_plus = nb_sp_plus;
                args_longest_run.nb_sp_minus = nb_sp_minus;
                args_longest_run.nb_su_plus = nb_su_plus;
                args_longest_run.nb_su_minus = nb_su_minus;
                args_longest_run.L = L;
                args_longest_run.U = U;
                args_longest_run.fairness_tolerence = fairness_tolerence;
                args_longest_run.TPp = TPp;
                args_longest_run.FPp = FPp;
                args_longest_run.TNp = TNp;
                args_longest_run.FNp = FNp;
                args_longest_run.TPu = TPu;
                args_longest_run.FPu = FPu;
                args_longest_run.TNu = TNu;
                args_longest_run.FNu = FNu;
            }
            if(!check_bounds.isFeasible()){ // no solution => the fairness constraint can never be satisfied using the current prefix -> we skip its evaluation without adding it to the queue
                improvedPruningCnt++;
                continue;
            }   
        }

        fairness_metrics fm = compute_fairness_metrics(cmg);
        
        switch (fairness)
        {
            case 1:
                unfairness = fm.statistical_parity;
                cmg.unfairnessLB = 0; //THIS CANCELS THE EFFECT OF THIS PART OF THE CODE FOR STATISTICAL PARITY (as we use CP now) -> delete line to use former pruning again
                break;
            case 2:
                unfairness = fm.predictive_parity;
                cmg.unfairnessLB = cmg.predparityLB; // lower bound is now for fairness metric nb 2
                break;
            case 3:
                unfairness = fm.predictive_equality;
                cmg.unfairnessLB = cmg.predequalityLB;
                break;
            case 4:
                unfairness = fm.equal_opportunity;
                cmg.unfairnessLB = cmg.equalOppLB;
                break;
            case 5:
                unfairness = fm.equalized_odds;
                cmg.unfairnessLB = 0;
                break;
            case 6:
                unfairness = fm.cond_use_acc_equality;
                cmg.unfairnessLB = 0;
                break;
            default:
                break;
        }
        if(useUnfairnessLB) {
            if(unfairness < cmg.unfairnessLB) { // should never happen ; if it happens we print useful information
                printf("PPV_min = %lf (in [%lf,%lf]), PPV_maj = %lf (in [%lf,%lf])\n",
                cmg.minority.nPPV, cmg.minority.min_ppv, cmg.minority.max_ppv, cmg.majority.nPPV, cmg.majority.min_ppv, cmg.majority.max_ppv);
                printf("TP_min = %d (in [%d,%d]), TP_maj = %d (in [%d,%d])\n",
                cmg.minority.nTP, cmg.minority.min_tp,cmg.minority.max_tp , cmg.majority.nTP, cmg.majority.min_tp, cmg.majority.max_tp);
                printf("FP_min = %d (in [%d,%d]), FP_maj = %d (in [%d,%d])\n",
                cmg.minority.nFP, cmg.minority.min_fp,cmg.minority.max_fp , cmg.majority.nFP, cmg.majority.min_fp, cmg.majority.max_fp);
                printf("FPR_min = %lf (in [%lf,%lf]), FPR_maj = %lf (in [%lf,%lf])\n",
                cmg.minority.nFPR, cmg.minority.min_fpr, cmg.minority.max_fpr, cmg.majority.nFPR, cmg.majority.min_fpr, cmg.majority.max_fpr);
                printf("TN_min = %d (in [%d,%d]), TN_maj = %d (in [%d,%d])\n",
                cmg.minority.nTN, cmg.minority.min_tn,cmg.minority.max_tn , cmg.majority.nTN, cmg.majority.min_tn, cmg.majority.max_tn);
                printf("FN_min = %d (in [%d,%d]), FN_maj = %d (in [%d,%d])\n",
                cmg.minority.nFN, cmg.minority.min_fn,cmg.minority.max_fn , cmg.majority.nFN, cmg.majority.min_fn, cmg.majority.max_fn);
                printf("FNR_min = %lf (in [%lf,%lf]), FNR_maj = %lf (in [%lf,%lf])\n",
                cmg.minority.nFNR, cmg.minority.min_fnr, cmg.minority.max_fnr, cmg.majority.nFNR, cmg.majority.min_fnr, cmg.majority.max_fnr);
            }
        }
        /* --- compute the objective function --- */
        if(mode == 2) { // Max fairness
            objective = unfairness + lower_bound;
        } else if(mode == 3) { // Espilon Constraint mode
            double misc = (double)(num_not_captured - default_correct) / nsamples;
            objective =  misc + lower_bound;
        } else if(mode == 4) { // Max accuracy 
            double misc = (double)(num_not_captured - default_correct) / nsamples;
            objective =  misc + lower_bound;
        } else { // Regularized mode
            double misc = (double)(num_not_captured - default_correct) / nsamples;
            /* Distance-to-a-reference objective function */
            /*double unfairnessObjective = 0.0;
            double accuracyObjective = 1.0;
            double distance = sqrt((beta*squareCalc(unfairness - unfairnessObjective)) + ((1-beta)*squareCalc((1 - misc) - accuracyObjective)));
            objective = distance + lower_bound;*/
            /* Weighted sum of objective functions */
            //objective = distance + lower_bound;
            objective =  (1-beta)*misc + beta*unfairness + lower_bound;
        }
        /* --- unfairness lower bound */
        /*double fairnesslb = 1.0;
        if(mode == 3) {
            if(useUnfairnessLB) {
                fairnesslb = 1 - cmg.unfairnessLB;
                if(fairnesslb < min_fairness_acceptable)
                    pruningCnt++;
            }
        }*/
        logger->addToObjTime(time_diff(t2));
        logger->incObjNum();
        if (objective < tree->min_objective()) {
            if(mode == 3) { // if mode 3 we check if the constraint on fairness is satisfied
                if((1-unfairness) > min_fairness_acceptable) {
                    best_rl_length = len_prefix;
                    if(debug) {
                    printf("min(objectivee): %1.5f -> %1.5f, length: %d (check -> %d), cache size: %zu, explored %lu nodes, pushed %d nodes (opt pruning = %d/%d), arriveHere = %d, permBound = %d.\n",
                    tree->min_objective(), objective, len_prefix, best_rl_length, tree->num_nodes(), exploredNodes, pushingTicket, improvedPruningCnt, improvedPruningCntTot, arriveHere, permBound);
                    //printf("(1-unfairness) = %lf, min_fairness_acceptable = %lf, fairnessLB=%lf\n",(1-unfairness),min_fairness_acceptable,fairnesslb);
                    //printf("TPmaj=%d, FPmaj=%d, TNmaj=%d, FNmaj=%d, TPmin=%d, FPmin=%d, TNmin=%d, FNmin=%d\n", cmg.majority.nTP,cmg.majority.nFP,cmg.majority.nTN,cmg.majority.nFN,cmg.minority.nTP,cmg.minority.nFP,cmg.minority.nTN,cmg.minority.nFN);
                    //printf("explored %d nodes before best solution.\n", exploredNodes);
                    }
                    nodesBeforeBest = exploredNodes;
                    cacheBeforeBest = tree->num_nodes();      
                    logger->setTreeMinObj(objective);
                    tree->update_min_objective(objective);
                    tree->update_opt_rulelist(parent_prefix, i);
                    tree->update_opt_predictions(parent, prediction, default_prediction);
                    logger->dumpState();                             
                }
            } else {                
                //printf("min(objectivee): %1.5f -> %1.5f, length: %d, cache size: %zu\n",
                //tree->min_objective(), objective, len_prefix, tree->num_nodes());
                //printf("explored %d nodes before best solution.\n", exploredNodes);
                logger->setTreeMinObj(objective);
                tree->update_min_objective(objective);
                tree->update_opt_rulelist(parent_prefix, i);
                tree->update_opt_predictions(parent, prediction, default_prediction);
                // dump state when min objective is updated
                logger->dumpState();
            }
        }
        // calculate equivalent points bound to capture the fact that the minority points can never be captured correctly
        if (tree->has_minority()) {
            rule_vand(not_captured_equivalent, not_captured, tree->minority(0).truthtable, nsamples, &num_not_captured_equivalent);
            equivalent_minority = (double)(num_not_captured_equivalent) / nsamples;
            lower_bound += equivalent_minority;
        }
        if (tree->ablation() != 2 && tree->ablation() != 3)
            lookahead_bound = lower_bound + c;
        else
            lookahead_bound = lower_bound;
        // only add node to our datastructures if its children will be viable
        if ((lookahead_bound < tree->min_objective()) && !pass) { //&& (fairnesslb>=min_fairness_acceptable)
            arriveHere++;
            double t3 = timestamp();
            // check permutation bound
            Node* n = p->insert(i, nrules, prediction, default_prediction,
                                   lower_bound, objective, parent, num_not_captured, nsamples,
                                   len_prefix, c, equivalent_minority, tree, not_captured, parent_prefix);
            logger->addToPermMapInsertionTime(time_diff(t3));
            // n is NULL if this rule fails the permutaiton bound
            if (n) {
                pushingTicket++;
                n->set_num(pushingTicket);
                double t4 = timestamp();
                tree->insert(n);
                logger->incTreeInsertionNum();
                logger->incPrefixLen(len_prefix);
                logger->addToTreeInsertionTime(time_diff(t4));
                double t5 = timestamp();
                q->push(n);
                logger->setQueueSize(q->size());
                if (tree->calculate_size())
                    logger->addQueueElement(len_prefix, lower_bound, false);
                logger->addToQueueInsertionTime(time_diff(t5));
            } else {
                permBound++;
            }
        } // else:  objective lower bound with one-step lookahead
    }
    rule_vfree(&captured_it);
    rule_vfree(&not_captured_yet);
    rule_vfree(&preds_prefix);
    rule_vfree(&captured_zeros_j);


    rule_vfree(&captured);
    rule_vfree(&captured_zeros);
    rule_vfree(&not_captured);
    rule_vfree(&not_captured_zeros);
    rule_vfree(&not_captured_equivalent);

    logger->addToRuleEvalTime(time_diff(t0));
    logger->incRuleEvalNum();
    logger->decPrefixLen(parent->depth());
    if (tree->calculate_size())
        logger->removeQueueElement(len_prefix - 1, parent_lower_bound, false);
    if (parent->num_children() == 0) {
        tree->prune_up(parent);
    } else {
        parent->set_done();
        tree->increment_num_evaluated();
    }
}

static size_t num_iter = 0;
static double min_objective = 0.0;
static VECTOR captured, not_captured;
static double start = 0.0;

/*
 * Explores the search space by using a queue to order the search process.
 * The queue can be ordered by DFS, BFS, or an alternative priority metric (e.g. lower bound).
 */
void bbound_begin(CacheTree* tree, Queue* q) {
    start = timestamp();
    num_iter = 0;
    rule_vinit(tree->nsamples(), &captured);
    rule_vinit(tree->nsamples(), &not_captured);

    logger->setInitialTime(start);
    logger->initializeState(tree->calculate_size());
    // initial log record
    logger->dumpState();         

    min_objective = 1.0;
    tree->insert_root();
    logger->incTreeInsertionNum();
    q->push(tree->root());
    logger->setQueueSize(q->size());
    logger->incPrefixLen(0);
    // log record for empty rule list
    logger->dumpState();
}

void bbound_loop(CacheTree* tree, 
                Queue* q, 
                PermutationMap* p,
                double beta,
                int fairness,
                rule_t* maj_v,
                rule_t* min_v,
                int mode,
                bool useUnfairnessLB,
                double min_fairness_acceptable,
                int kBest,
                bool forbidSensAttr,
                double accuracyUpperBound){

    double t0 = timestamp();
    int verbosity = logger->getVerbosity();
    size_t queue_min_length = logger->getQueueMinLen();
    int cnt;
    std::pair<Node*, tracking_vector<unsigned short, DataStruct::Tree> > node_ordered = q->select(kBest, tree, captured);
    logger->addToNodeSelectTime(time_diff(t0));
    logger->incNodeSelectNum();
    if (node_ordered.first) {
        double t1 = timestamp();
        // not_captured = default rule truthtable & ~ captured
        rule_vandnot(not_captured,
                     tree->rule(0).truthtable, captured,
                     tree->nsamples(), &cnt);
        evaluate_children(tree, node_ordered.first, node_ordered.second, not_captured, q, p, beta, fairness, maj_v, min_v, mode, useUnfairnessLB,
                        min_fairness_acceptable, forbidSensAttr, accuracyUpperBound);
        logger->addToEvalChildrenTime(time_diff(t1));
        logger->incEvalChildrenNum();

        if (tree->min_objective() < min_objective) {
            min_objective = tree->min_objective();
            if (verbosity >= 10)
                printf("before garbage_collect. num_nodes: %zu\n", tree->num_nodes());
            logger->dumpState();
            tree->garbage_collect();
            logger->dumpState();
            if (verbosity >= 10)
                printf("after garbage_collect. num_nodes: %zu\n", tree->num_nodes());
        }
    }
    logger->setQueueSize(q->size());
    if (queue_min_length < logger->getQueueMinLen()) {
        // garbage collect the permutation map: can be simplified for the case of BFS
        queue_min_length = logger->getQueueMinLen();
        //pmap_garbage_collect(p, queue_min_length);
    }
    ++num_iter;
    if ((num_iter % 10000) == 0) {
        if (verbosity >= 10)
            printf("iter: %zu, tree: %zu, queue: %zu, pmap: %zu, time elapsed: %f\n",
                   num_iter, tree->num_nodes(), q->size(), p->size(), time_diff(start));
    }
    if ((num_iter % logger->getFrequency()) == 0) {
        // want ~1000 records for detailed figures
        logger->dumpState();
    }
}
    
std::vector<unsigned long> bbound_end(CacheTree* tree, Queue* q, PermutationMap* p, bool early, rule_t* rules, rule_t* labels) {
    int verbosity = logger->getVerbosity();
    bool print_queue = 0;
    if(debug) {
        printf("explored %lu nodes.\n", exploredNodes);
        printf("using new filtering pruned %d/%d nodes.\n", improvedPruningCnt, improvedPruningCntTot);
        printf("Total solving time = %Lf s\n", total_solving_time/1000000.0);
        printf("Longest fitlering run took %f ms.\n", longestfilteringrun/1000.0);
        printf("Average time per solver call = %Lf ms\n", (total_solving_time/1000.0)/total_solver_calls);
        printf("%d/%f solver calls timed out.\n", timeoutCnt, total_solver_calls);
        printf("Number of nodes in the trie at exit : %d\n",  tree->num_nodes());
        printf("params : (%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d,%d,%d,%d,%d)\n",args_longest_run.nb_sp_plus,
                    args_longest_run.nb_sp_minus,
                    args_longest_run.nb_su_plus,
                    args_longest_run.nb_su_minus,
                    args_longest_run.L,
                    args_longest_run.U,
                    args_longest_run.fairness_tolerence,
                    args_longest_run.TPp,
                    args_longest_run.FPp,
                    args_longest_run.TNp,
                    args_longest_run.FNp,
                    args_longest_run.TPu,
                    args_longest_run.FPu,
                    args_longest_run.TNu,
                    args_longest_run.FNu);
    }
    improvedPruningCnt = 0;
    improvedPruningCntTot = 0;
    longestfilteringrun = -1.0;
    total_solving_time = 0.0;
    total_solver_calls = 0.0;
    max_depth = 0;
    pushingTicket = 0;
    pruningCnt = 0;
    best_rl_length = 0;
    exploredNodes = 0;
    firstPass = true;
    firstPass2 = true;
    firstCall = true;
    logger->dumpState(); // second last log record (before queue elements deleted)
   // if(pruningCnt > 0)
        //printf("Pruned %d subtrees with unfairness lower bound.\n", pruningCnt);
    if (verbosity >= 5)
        printf("iter: %zu, tree: %zu, queue: %zu, pmap: %zu, time elapsed: %f\n",
               num_iter, tree->num_nodes(), q->size(), p->size(), time_diff(start));
    
    if (!early) {
        if (q->empty()) {
            if (verbosity >= 1) 
                printf("Exited because queue empty\n");
        }
        else if (verbosity >= 1)
            printf("Exited because max number of nodes in the tree was reached\n");
    }

    // Print out queue
    ofstream f;
    if (print_queue) {
        char fname[] = "queue.txt";
        if (verbosity >= 1) {
            printf("Writing queue elements to: %s\n", fname);
        }
        f.open(fname, ios::out | ios::trunc);
        f << "lower_bound objective length frac_captured rule_list\n";
    }

    // Clean up data structures
    if (verbosity >= 1) {
        printf("Deleting queue elements and corresponding nodes in the cache,"
            "since they may not be reachable by the tree's destructor\n");
        printf("\nminimum objective: %1.10f\n", tree->min_objective());
    }
    Node* node;
    double min_lower_bound = 1.0;
    double lb;
    size_t num = 0;
    while (!q->empty()) {
        node = q->front();
        q->pop();
        if (node->deleted()) {
            tree->decrement_num_nodes();
            logger->removeFromMemory(sizeof(*node), DataStruct::Tree);
            delete node;
        } else {
            lb = node->lower_bound() + tree->c();
            if (lb < min_lower_bound)
                min_lower_bound = lb;
            if (print_queue) {
                std::pair<tracking_vector<unsigned short, DataStruct::Tree>, tracking_vector<bool, DataStruct::Tree> > pp_pair = node->get_prefix_and_predictions();
                tracking_vector<unsigned short, DataStruct::Tree> prefix = std::move(pp_pair.first);
                tracking_vector<bool, DataStruct::Tree> predictions = std::move(pp_pair.second);
                f << node->lower_bound() << " " << node->objective() << " " << node->depth() << " "
                  << (double) node->num_captured() / (double) tree->nsamples() << " ";
                for(size_t i = 0; i < prefix.size(); ++i) {
                    f << tree->rule_features(prefix[i]) << "~"
                      << predictions[i] << ";";
                }
                f << "default~" << predictions.back() << "\n";
                num++;
            }
        }
    }
    
    /* Compute confidence scores and exact accuracy */
    compData res = computeFinalFairness(tree->nsamples(), 
                         tree->opt_rulelist(), 
                         tree->opt_predictions(),
                         rules,
                         labels);
    
    tree->setConfScores(res.conf_scores);
    tree->setFinalAcc(res.accuracy);
    if (verbosity >= 1)
        printf("minimum lower bound in queue: %1.10f\n\n", min_lower_bound);
    
    if (print_queue)
        f.close();
    // last log record (before cache deleted)
    logger->dumpState();

    rule_vfree(&captured);
    rule_vfree(&not_captured);
    std::vector<unsigned long> returnVal;
    returnVal.push_back(nodesBeforeBest);
    returnVal.push_back(cacheBeforeBest);

    return returnVal;
}
