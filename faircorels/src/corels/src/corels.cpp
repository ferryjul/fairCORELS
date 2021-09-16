#include "queue.hh"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "filtering_algorithms.cpp"
//#include <time.h> // for cp filtering running time measures
#include "../../../_corels.h" // for opt pruning
#include <ctime> // for solver RT measurements
#include <numeric> // for solver RT measurements

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
int filtering_modeG = -1;
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
// for new upper bound (pruning)
rule_t* incons_min_errs;
VECTOR incons_remaining;
int Gupper_bound_filtering = 0;
int U_improved = 0;
std::vector<double> timesTot; 
// -----------------------------

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

// reduced version, working slightly faster, for prefix only (used for PPC filtering variables init)
confusion_matrix_groups compute_confusion_matrix_prefix(VECTOR parent_prefix_predictions,
                                                CacheTree* tree,
                                                VECTOR parent_not_captured, 
                                                rule_t* maj_v,
                                                rule_t* min_v,
                                                bool prediction, 
                                                bool default_prediction){

    // datastructures to store the results
    confusion_matrix_groups cmg;
    confusion_matrix cm_minority;
    confusion_matrix cm_majority;
    int nsamples = tree->nsamples();
    int pm;

    // true positives, false negatives, true negatives, and false positives
    VECTOR TP, FP, FN, TN;
    rule_vinit(nsamples, &TP);
    rule_vinit(nsamples, &FP);
    rule_vinit(nsamples, &FN);
    rule_vinit(nsamples, &TN);

    rule_vand(TP, parent_prefix_predictions, tree->label(1).truthtable, nsamples, &pm);
    rule_vand(FP, parent_prefix_predictions, tree->label(0).truthtable, nsamples, &pm);
    rule_vandnot(FN, tree->label(1).truthtable, parent_prefix_predictions, nsamples, &pm);
    rule_vandnot(TN, tree->label(0).truthtable, parent_prefix_predictions, nsamples, &pm);

    // restrict to instances captured by prefix
    rule_vandnot(TP, TP, parent_not_captured, nsamples, &pm);
    rule_vandnot(FP, FP, parent_not_captured, nsamples, &pm);
    rule_vandnot(FN, FN, parent_not_captured, nsamples, &pm);
    rule_vandnot(TN, TN, parent_not_captured, nsamples, &pm);

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


    cmg.minority.nTP = nTP_min;
    cmg.majority.nTP = nTP_maj;
    cmg.minority.nFP = nFP_min;
    cmg.majority.nFP = nFP_maj;
    cmg.minority.nTN = nTN_min;
    cmg.majority.nTN = nTN_maj;
    cmg.minority.nFN = nFN_min;
    cmg.majority.nFN = nFN_maj;
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

    return cmg;
}

int countUncapturedIncons(VECTOR parent_not_captured, int nsamples){
    int ret = 0;

    rule_vand(incons_remaining, parent_not_captured, incons_min_errs[0].truthtable, nsamples, &ret);
    //if(count_ones_vector(incons_remaining, nsamples)!=ret){
    //    printf("count_ones_vector(incons_remaining, nsamples)=%d, ret=%d\n", count_ones_vector(incons_remaining, nsamples), ret);
    //    printf("abnormal, exiting\n");
    //    exit(-1);
    //}
    return ret;
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


    // restrict to instances captured by prefix
    int nminTP_min, nminFP_min, nminFN_min, nminTN_min;
    rule_vandnot(TP_min, TP_min, not_captured, nsamples, &nminTP_min);
    rule_vandnot(FP_min, FP_min, not_captured, nsamples, &nminFP_min);
    rule_vandnot(FN_min, FN_min, not_captured, nsamples, &nminFN_min);
    rule_vandnot(TN_min, TN_min, not_captured, nsamples, &nminTN_min);

    cm_minority.nminTP = nminTP_min;
    cm_minority.nminFP = nminFP_min;
    cm_minority.nminFN = nminFN_min;
    cm_minority.nminTN = nminTN_min;

    int nminTP_maj, nminFP_maj, nminFN_maj, nminTN_maj;
    rule_vandnot(TP_maj, TP_maj, not_captured, nsamples, &nminTP_maj);
    rule_vandnot(FP_maj, FP_maj, not_captured, nsamples, &nminFP_maj);
    rule_vandnot(FN_maj, FN_maj, not_captured, nsamples, &nminFN_maj);
    rule_vandnot(TN_maj, TN_maj, not_captured, nsamples, &nminTN_maj);

    cm_majority.nminTP = nminTP_maj;
    cm_majority.nminFP = nminFP_maj;
    cm_majority.nminFN = nminFN_maj;
    cm_majority.nminTN = nminTN_maj;

    cmg.majority = cm_majority;
    cmg.minority = cm_minority;

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
                        int filteringMode,
                        double min_fairness_acceptable,
                        bool forbidSensAttr,
                        double accuracyUpperBound){
    
    if(firstCall){
        filtering_modeG = filteringMode;
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
        if(Gupper_bound_filtering==0){
                std::cout << "Using old U computation" << std::endl;
            } else{
                std::cout << "Using new U computation" << std::endl;
            }
        if(debug) {
            
            if(fairness == 1 && filteringMode)
                printf("will perform improved SP pruning\n");
            else if(fairness == 2 && filteringMode)
                printf("will perform improved PP pruning\n");
            else if(fairness == 3 && filteringMode)
                printf("will perform improved PE pruning\n");
            else if(fairness == 4 && filteringMode)
                printf("will perform improved EO pruning\n");
            else if(fairness == 5 && filteringMode)
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
    VECTOR captured_it, not_captured_yet, captured_zeros_j, preds_prefix, captured_prefix;

    int nb, nb2, pm;

    rule_vinit(tree->nsamples(), &captured_it);
    rule_vinit(tree->nsamples(), &not_captured_yet);
    rule_vinit(tree->nsamples(), &preds_prefix);
    rule_vinit(tree->nsamples(), &captured_zeros_j);
    //rule_vinit(tree->nsamples(), &captured_prefix);
    // Initially not_captured_yet is full of ones
    rule_vor(not_captured_yet, tree->label(0).truthtable, tree->label(1).truthtable, tree->nsamples(),&nb);

    // Initially preds_prefix is full of zeros
    rule_vclear(tree->nsamples(), preds_prefix);
    //rule_vclear(tree->nsamples(), captured_prefix);

    int depth = len_prefix;
    tracking_vector<unsigned short, DataStruct::Tree>::iterator it;

    /*bool prefixMatched = true;
    int nbRules = 0;*/
    //printf("Prefix:\n");
    for (it = parent_prefix.begin(); it != parent_prefix.end(); it++) {
        //rule_vor(captured_prefix, captured_prefix, tree->rule(*it).truthtable, tree->nsamples(), &nb);
        rule_vand(captured_it, not_captured_yet, tree->rule(*it).truthtable, tree->nsamples(), &nb);
        rule_vandnot(not_captured_yet, not_captured_yet, captured_it, tree->nsamples(), &pm);
        rule_vand(captured_zeros_j, captured_it, tree->label(0).truthtable, tree->nsamples(), &nb2);
       // nbRules++;
        if(nb2 <= (nb - nb2)) { //then prediction is 1
            rule_vor(preds_prefix, preds_prefix, captured_it, tree->nsamples(), &nb);
        }
    /*    if(nbRules == 1){
            //printf("1: %s\n", tree->rule(*it).features);
            if( strcmp(tree->rule(*it).features, "sex_Female")){
                prefixMatched = false;
            }
        }else if(nbRules == 2){
            //printf("2: %s\n", tree->rule(*it).features);
            if( strcmp(tree->rule(*it).features, "age_high")){
                prefixMatched = false;
            }
        }else if(nbRules == 3){
            //printf("3: %s\n", tree->rule(*it).features);
                if( strcmp(tree->rule(*it).features, "priors_count_low")){
                prefixMatched = false;
            }
        }else if(nbRules == 4){
           // printf("4: %s\n", tree->rule(*it).features);
            if( strcmp(tree->rule(*it).features, "not_age_middle")){
                prefixMatched = false;
            }
        }*/
    }
    //printf("Prefix captured %d instances.\n", tree->nsamples()-count_ones_vector(not_captured_yet, tree->nsamples()));
    //printf("There are %d inconsistent examples in minority (=> min errors)\n", count_ones_vector(incons_min_errs->truthtable, tree->nsamples()));
    //int tot = 0;
   
    //printf("There were %d bits 1 in parent_not_captured.\n", tot);
  /*  if(nbRules != 3){
        prefixMatched = false;
    }
    if(prefixMatched){
        printf("found prefix!\n");
        for (it = parent_prefix.begin(); it != parent_prefix.end(); it++) {
            printf("%s\n", tree->rule(*it).features);
        }
    }*/
    /*int a, b;
    a = count_ones_vector(captured_prefix, tree->nsamples());
    b =  tree->nsamples() - count_ones_vector(parent_not_captured, tree->nsamples());

    if (a != b){
        printf("count differ: a = %d, b = %d\n", a, b);
    } */
    bool prefixPassedCP = true;
    if((filteringMode == 1 || filteringMode == 3)&& best_rl_length > 0 && (fairness == 1 || fairness == 3 || fairness == 4 || fairness == 5)){  // Here occurs the PPC Filtering
			int L = (1 - (tree->min_objective()  - (len_prefix*c) ) )*tree->nsamples();
            // old, wrong computation : (1 - (tree->min_objective() + ((best_rl_length-len_prefix)*c)))*tree->nsamples(); // (1 - misc)*nb_samples = nb inst well classif by current best model
			
			
            float fairness_tolerence = 1-min_fairness_acceptable; // equiv max unfairness acceptable

            confusion_matrix_groups cmg = compute_confusion_matrix_prefix(preds_prefix, tree, parent_not_captured,
                                                                                 maj_v, min_v, prediction, default_prediction);

			int TPp = cmg.minority.nTP;
			int FPp = cmg.minority.nFP;
			int TNp = cmg.minority.nTN;
			int FNp = cmg.minority.nFN;
			int TPu = cmg.majority.nTP;
			int FPu = cmg.majority.nFP;
			int TNu = cmg.majority.nTN;
			int FNu = cmg.majority.nFN;
            int U;
            // old, fixed but easy, cheap and safe computation: int U = accuracyUpperBound * (tree->nsamples());
            if(Gupper_bound_filtering==0){
                U = accuracyUpperBound * (tree->nsamples());
            }else{
                int remainingInconsErrors = countUncapturedIncons(parent_not_captured, tree->nsamples());
                U =   tree->nsamples() - (remainingInconsErrors + FNp + FPp + FNu + FPu);
                int oldU = accuracyUpperBound * (tree->nsamples());
                if(oldU>U){
                    U_improved++;
                }else if(oldU<U){
                    std::cout << "oldU = " << oldU << ", new U = " << U << std::endl;
                    exit(-1);
                }
                
            }
            

            int config = 0;
            if(fairness == 1){
                config = 8;
            } else if(fairness == 4){
                config = 2;
            }
            //FilteringStatisticalParity check_bounds(nb_sp_plus,nb_sp_minus, nb_su_plus, nb_su_minus, L,U , fairness_tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
            //check_bounds.run(0, 0);
            double maxSolvingTime = 5*10e9; // <- 5 seconds is already a lot, it simply helps avoiding to get stuck
            struct runResult res = runFiltering(fairness, //metric
                                config, //solver config
                                nb_sp_plus,nb_sp_minus, 
                                nb_su_plus, nb_su_minus, 
                                L,U , 
                                fairness_tolerence, 
                                TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu,
                                maxSolvingTime //timeout (nanoseconds, or -1 for no timeout)
                                );

            if(res.result == UNSAT){ // no solution => the fairness constraint can never be satisfied using the current prefix -> we skip its evaluation without adding it to the queue
                improvedPruningCnt++;
                prefixPassedCP = false;
            }   
    }

    // begin evaluating children
    bool pass;
    for (i = 1; prefixPassedCP && i < nrules; i++) {
        pass = false;
     /*   if(prefixMatched){
             printf("Working on RL: %s\n", tree->rule(i).features);
        }*/
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
       /* if(prefixMatched){
             printf("Working on RL(2): %s\n", tree->rule(i).features);
        }
        if(! strcmp(tree->rule(i).features, "not_age_middle") && prefixMatched){
            printf("Working on RL\n");
        }*/
        bool filteringOK = true;
        int res_opt = tree->nsamples();

        // && best_rl_length > 0 <- Was also in condition of the 'if' below !
        if((filteringMode == 2 || filteringMode == 3 || filteringMode == 4)  && (fairness == 1 || fairness == 3 || fairness == 4 || fairness == 5)){  // Here occurs the PPC Filtering
            int L = (1 - (tree->min_objective()  - ((len_prefix+1)*c) ) )*tree->nsamples();
            // Filtering performed to know whether extension will be a viable prefix, hence the +1
            // Note that solver can say UNSAT for chlidren and RL meet the fairness constraint
            // old, wrong computation : (1 - (tree->min_objective() + ((best_rl_length-len_prefix)*c)))*tree->nsamples(); // (1 - misc)*nb_samples = nb inst well classif by current best model
			// int U = accuracyUpperBound * (tree->nsamples());
			float fairness_tolerence = 1-min_fairness_acceptable; // equiv max unfairness acceptable

			int TPp = cmg.minority.nminTP;
			int FPp = cmg.minority.nminFP;
			int TNp = cmg.minority.nminTN;
			int FNp = cmg.minority.nminFN;
			int TPu = cmg.majority.nminTP;
			int FPu = cmg.majority.nminFP;
			int TNu = cmg.majority.nminTN;
			int FNu = cmg.majority.nminFN;

            // old, fixed but easy, cheap and safe computation: int U = accuracyUpperBound * (tree->nsamples());
            int U;
            if(Gupper_bound_filtering==0){
                U = accuracyUpperBound * (tree->nsamples());
            }else{
                int remainingInconsErrors = countUncapturedIncons(not_captured, tree->nsamples());
                U =   tree->nsamples() - (remainingInconsErrors + FNp + FPp + FNu + FPu);
                int oldU = accuracyUpperBound * (tree->nsamples());
                if(oldU>U){
                    U_improved++;
                }else if(oldU<U){
                    std::cout << "oldU = " << oldU << ", new U = " << U << std::endl;
                    exit(-1);
                }
                
            }

            // print all parameters provided to solver
            /*std::cout << "---------------------------------------" << std::endl;
            std::cout << "L = " << L << ";" << std::endl;
            std::cout << "U = " << U << ";" << std::endl;
            std::cout << "nb_sp_plus = " << nb_sp_plus << ";" << std::endl;
            std::cout << "nb_sp_minus = " << nb_sp_minus << ";" << std::endl;
            std::cout << "nb_su_plus = " << nb_su_plus << ";" << std::endl;
            std::cout << "nb_su_minus = " << nb_su_minus << ";" << std::endl;
            std::cout << "TPp = " << TPp << ";" << std::endl;
            std::cout << "FPp = " << FPp << ";" << std::endl;
            std::cout << "TNp = " << TNp << ";" << std::endl;
            std::cout << "FNp = " << FNp << ";" << std::endl;
            std::cout << "TPu = " << TPu << ";" << std::endl;
            std::cout << "FPu = " << FPu << ";" << std::endl;
            std::cout << "TNu = " << TNu << ";" << std::endl;
            std::cout << "FNu = " << FNu << ";" << std::endl;
            std::cout << "tolerence = " << fairness_tolerence << ";" << std::endl;*/
            // ---------------------------------------

            if(filteringMode == 2 || filteringMode == 3){

                std::clock_t start = std::clock();

                int config = 0;
                if(fairness == 1){
                    config = 8;
                } else if(fairness == 4){
                    config = 2;
                }
                double maxSolvingTime = 5*10e9; // <- 5 seconds is already a lot, it simply helps avoiding to get stuck
                struct runResult res = runFiltering(fairness, //metric
                                    config, //solver config
                                    nb_sp_plus,nb_sp_minus, 
                                    nb_su_plus, nb_su_minus, 
                                    L,U , 
                                    fairness_tolerence, 
                                    TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu,
                                    maxSolvingTime //timeout (nanoseconds, or -1 for no timeout)
                                    );

                if(res.result == UNSAT){ // no solution => the fairness constraint can never be satisfied using the current prefix -> we skip its evaluation without adding it to the queue
                    improvedPruningCnt++;

                    filteringOK = false;
                }   

                std::clock_t end = std::clock();
                double cpu_time_used_microsecs = ((double) (end - start) * 1000000) / CLOCKS_PER_SEC;
                timesTot.push_back(cpu_time_used_microsecs);
            } else if(filteringMode == 4){
                std::clock_t start = std::clock();
                res_opt = perform_opt_pruning_wrapper(nb_sp_plus, nb_sp_minus, nb_su_plus, nb_su_minus, L, U, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu);
                if(res_opt < 0){ // no solution => the fairness constraint can never be satisfied using the current prefix -> we skip its evaluation without adding it to the queue
                    improvedPruningCnt++;
                    filteringOK = false;
                }   
                std::clock_t end = std::clock();
                double cpu_time_used_microsecs = ((double) (end - start) * 1000000) / CLOCKS_PER_SEC;
                timesTot.push_back(cpu_time_used_microsecs);
            }
            
    }

        fairness_metrics fm = compute_fairness_metrics(cmg);
        
        switch (fairness)
        {
            case 1:
                unfairness = fm.statistical_parity;
                //cmg.unfairnessLB = 0; // cancels the effect of the simple bound, now useless as we use improved cp filtering
                break;
            case 2:
                unfairness = fm.predictive_parity;
                //cmg.unfairnessLB = cmg.predparityLB;
                break;
            case 3:
                unfairness = fm.predictive_equality;
                //cmg.unfairnessLB = cmg.predequalityLB;
                break;
            case 4:
                unfairness = fm.equal_opportunity;
                //cmg.unfairnessLB = 0; // cancels the effect of the simple bound, now useless as we use improved cp filtering
                break;
            case 5:
                unfairness = fm.equalized_odds;
                //cmg.unfairnessLB = 0; 
                break;
            case 6:
                unfairness = fm.cond_use_acc_equality;
                //cmg.unfairnessLB = 0;
                break;
            default:
                break;
        }
        /*if(useUnfairnessLB) {
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
        }*/
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
                    //if(debug) {
                    printf("min(objectivee): %1.5f -> %1.5f, length: %d (check -> %d), cache size: %zu, explored %lu nodes, pushed %d nodes (opt pruning = %d/%d), arriveHere = %d, permBound = %d, parent bound opt was: %lf.\n",
                    tree->min_objective(), objective, len_prefix, best_rl_length, tree->num_nodes(), exploredNodes, pushingTicket, improvedPruningCnt, improvedPruningCntTot, arriveHere, permBound, parent->get_opt_bound());
                    //printf("(1-unfairness) = %lf, min_fairness_acceptable = %lf, fairnessLB=%lf\n",(1-unfairness),min_fairness_acceptable,fairnesslb);
                    //printf("TPmaj=%d, FPmaj=%d, TNmaj=%d, FNmaj=%d, TPmin=%d, FPmin=%d, TNmin=%d, FNmin=%d\n", cmg.majority.nTP,cmg.majority.nFP,cmg.majority.nTN,cmg.majority.nFN,cmg.minority.nTP,cmg.minority.nFP,cmg.minority.nTN,cmg.minority.nFN);
                    //printf("explored %d nodes before best solution.\n", exploredNodes);
                    //}
                    nodesBeforeBest = exploredNodes;
                    cacheBeforeBest = tree->num_nodes();      
                    logger->setTreeMinObj(objective);
                    tree->update_min_objective(objective);
                    tree->update_opt_rulelist(parent_prefix, i);
                    tree->update_opt_predictions(parent, prediction, default_prediction);
                    logger->dumpState();      
                    /*printf("To improve objective function value, must classify correctly #examples: (lambda = %lf)\n", c);
                    printf("(Note that current best objective is %lf, for RL with length %d.\n", tree->min_objective(), best_rl_length);
                    int test =  (1 - (tree->min_objective()  - c))*tree->nsamples();
                    printf("%d/%d for RL of length 1.\n", test, tree->nsamples());
                    test =  (1 - (tree->min_objective()  - (2*c)))*tree->nsamples();
                    printf("%d/%d for RL of length 2.\n", test, tree->nsamples());
                    test =  (1 - (tree->min_objective()  - (3*c)))*tree->nsamples();
                    int test2 = (1 - (tree->min_objective()  - ((4+1)*c) ) )*tree->nsamples();
                    printf("%d/%d for RL of length 3 (test2=%d).\n", test, tree->nsamples(), test2);
                    test =  (1 - (tree->min_objective()  - (4*c)))*tree->nsamples();
                    printf("%d/%d for RL of length 4.\n", test, tree->nsamples());
                    test =  (1 - (tree->min_objective()  - (5*c)))*tree->nsamples();
                    printf("%d/%d for RL of length 5.\n", test, tree->nsamples());         
                    int U = accuracyUpperBound * (tree->nsamples());
                    printf("--(upper bound is %d/%d)\n", U, tree->nsamples());   */           
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
        if(filteringOK == false){ // do not insert children nodes if filering not OK
            continue;
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
                n->set_unfairness(unfairness);
                // opt bound for queue ordering
                double opt_bound = ((len_prefix+1)*c)+(((double)tree->nsamples() - (double)res_opt)/(double)tree->nsamples());
                //if(opt_bound <= (len_prefix+2)*c ){
                //    std::cout << "opt bound is " << opt_bound << "(lambda part = " << ((len_prefix+1)*c) << ", error part = " << (((double)tree->nsamples() - (double)res_opt)/(double)tree->nsamples()) << ")" << std::endl;
                //    std::cout << "tree->nsamples()=" << tree->nsamples() << ", res_opt=" << res_opt << std::endl;
                //}
                n->set_opt_bound(opt_bound);
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
    //rule_vfree(&captured_prefix);
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
void bbound_begin(CacheTree* tree, Queue* q, rule_t* G_incons_min_errs, int upper_bound_filtering) {
    Gupper_bound_filtering = upper_bound_filtering;
    if(Gupper_bound_filtering > 0){
        rule_vinit(tree->nsamples(), &incons_remaining);
    }        

    incons_min_errs = G_incons_min_errs;
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
                int filteringMode,
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
        evaluate_children(tree, node_ordered.first, node_ordered.second, not_captured, q, p, beta, fairness, maj_v, min_v, mode, filteringMode,
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
    if(Gupper_bound_filtering > 0){
        std::cout << "Improved upper bound " << U_improved << "/" << improvedPruningCntTot << " times." << std::endl;
        rule_vfree(&incons_remaining);
    }   
    if(filtering_modeG == 4){
        print_memo_info_opt_pruning_auditor();
    }
    if(filtering_modeG == 2 || filtering_modeG == 4){
        std::cout << "Average solving time was " << std::accumulate(timesTot.begin(), timesTot.end(), 0)/timesTot.size() << " microsecs." << std::endl;
    }
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
