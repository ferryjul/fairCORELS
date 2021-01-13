#include <stdio.h>
#include <iostream>
#include <set>

#include "queue.hh"
#include "run.hh"

#define BUFSZ 512

NullLogger* logger = nullptr;
static PermutationMap* g_pmap = nullptr;
static CacheTree* g_tree = nullptr;
static Queue* g_queue = nullptr;
double g_init = 0.0;
std::set<std::string> g_verbosity;
rule_t * Grules;
rule_t * Glabels;
rule_t * Gmeta;
int SPolicy = -1;
int modeBFS = -1;
int map_type_arg = -1;
int currLimit = -1;
static CacheTree* best_tree = nullptr;
static Queue* best_queue = nullptr;
static PermutationMap* best_map = nullptr;
bool usedRestart = false;
bool first = true;
int nsamplesG = -1;
int nrulesG = -1;
int ablationG = -1;
int calculate_sizeG = -1;
double Gc = -1;
int *lubySeq;
int indLuby = -1;
bool forbidSensAttr = false;
int lubySeqSize = -1;
int callsNB = 1000;
int initCallsNB = 1000;
double accUpperBound;
rule_t* Gmaj_vect;
rule_t* Gmin_vect;
bool debugRun = false; // for printing more info while running/exploring

int run_corels_begin(double c, char* vstring, int curiosity_policy,
                  int map_type, int ablation, int calculate_size, int nrules, int nlabels,
                  int nsamples, rule_t* rules, rule_t* labels, rule_t* meta, int freq, 
                  char* log_fname, int BFSmode, int seed, bool forbidSensAttr_val, rule_t* maj_v, int nmaj_v,
                  rule_t* min_v, int nmin_v, double accuracy_upper_bound, int max_calls)
{
    callsNB = max_calls;
    initCallsNB = max_calls;
    if(debugRun) {
        printf("Will explore at most %d nodes.\n", callsNB);
        printf("Provided accuracy upper bound = %lf\n", accuracy_upper_bound);
    }
    
    accUpperBound = accuracy_upper_bound;
    // Check correctness
    if(nmaj_v != nmin_v){
        printf("Incorrect argument : nmaj and nmin should be equal\n");
        exit(-1);
    }
    int nbMaj = count_ones_vector(maj_v[1].truthtable, nmaj_v);
    int nbMin = count_ones_vector(min_v[1].truthtable, nmin_v);
    //printf("Maj vector : captures %d/%d instances.\n", nbMaj, nmaj_v);
    //printf("Min vector : captures %d/%d instances.\n", nbMin, nmin_v);
    if((nbMaj + nbMin) > nsamples){
        printf("Incorrect argument : majority and minority groups intersection should be empty.\n");
    }
    Gmaj_vect =  maj_v;
    Gmin_vect =  min_v;

    int nbMajG = count_ones_vector(Gmaj_vect[1].truthtable, nmaj_v);
    int nbMinG = count_ones_vector(Gmin_vect[1].truthtable, nmin_v);
    if((nbMajG + nbMinG) > nsamples){
        printf("Error : Internal error in vector copy \n");
    }
    //printf("Maj vector G : captures %d/%d instances.\n", nbMajG, nmaj_v);
    //printf("Min vector G : captures %d/%d instances.\n", nbMinG, nmin_v);

    forbidSensAttr = forbidSensAttr_val;
    Grules = rules;
    Glabels = labels;
    currLimit = -1;
    Gmeta = meta;
    Gc = c;
    SPolicy = curiosity_policy;
    nsamplesG = nsamples;
    nrulesG = nrules;
    ablationG = ablation;
    calculate_sizeG = calculate_size;
    //printf("seed = %d\n", seed);
    srand(seed);
    // Check arguments
    if(BFSmode < 0 || BFSmode > 4) {
        printf("Error : BFSmode should be in {1, 2, 3, 4}\n");
        exit(-1);
    }
    modeBFS = BFSmode;
    map_type_arg = map_type;
    g_verbosity.clear();

    const char *voptions = "rule|label|minor|samples|progress|loud";

    char *vopt = NULL;
    char *vcopy = m_strdup(vstring);
    char *vcopy_begin = vcopy;
    while ((vopt = m_strsep(&vcopy, ',')) != NULL) {
        if (!strstr(voptions, vopt)) {
            fprintf(stderr, "verbosity options must be one or more of (%s)\n", voptions);
            return -1;
        }
        g_verbosity.insert(vopt);
    }
    free(vcopy_begin);

    if (g_verbosity.count("loud")) {
        g_verbosity.insert("progress");
        g_verbosity.insert("label");
        g_verbosity.insert("rule");
        g_verbosity.insert("minor");
    }

#ifndef GMP
    if (g_verbosity.count("progress"))
        printf("**Not using GMP library**\n");
#endif

    if (g_verbosity.count("rule")) {
        printf("%d rules %d samples\n\n", nrules, nsamples);
        rule_print_all(rules, nrules, nsamples, g_verbosity.count("samples"));
        printf("\n\n");
    }

    if (g_verbosity.count("label")) {
        printf("Labels (%d) for %d samples\n\n", nlabels, nsamples);
        rule_print_all(labels, nlabels, nsamples, g_verbosity.count("samples"));
        printf("\n\n");
    }

    if (g_verbosity.count("minor") && meta) {
        printf("Minority bound for %d samples\n\n", nsamples);
        rule_print_all(meta, 1, nsamples, g_verbosity.count("samples"));
        printf("\n\n");
    }

    if (g_tree)
        delete g_tree;
    g_tree = nullptr;

    if (g_queue)
        delete g_queue;
    g_queue = nullptr;

    if (g_pmap)
        delete g_pmap;
    g_pmap = nullptr;

    int v = 0;
    if (g_verbosity.count("loud"))
        v = 1000;
    else if (g_verbosity.count("progress"))
        v = 1;

    if(!logger) {
        if(log_fname)
            logger = new Logger(c, nrules, v, log_fname, freq);
        else {
            logger = new PyLogger();
            logger->setVerbosity(v);
        }
    }

    g_init = timestamp();
    char run_type[BUFSZ];
    strcpy(run_type, "LEARNING RULE LIST via ");
    char const *type = "node";
    if (curiosity_policy == 1) {
        strcat(run_type, "CURIOUS");
        g_queue = new Queue(curious_cmp, run_type);
        type = "curious";
    } else if (curiosity_policy == 2) {
        strcat(run_type, "LOWER BOUND");
        g_queue = new Queue(lb_cmp, run_type);
    } else if (curiosity_policy == 3) {
        strcat(run_type, "OBJECTIVE");
        g_queue = new Queue(objective_cmp, run_type);
    } else if (curiosity_policy == 4) {
        strcat(run_type, "DFS");
        g_queue = new Queue(dfs_cmp, run_type);
    } else {
        strcat(run_type, "BFS");
        switch(BFSmode) {
            case 0:
                g_queue = new Queue(base_cmp, run_type);
                break;
            case 1:
                g_queue = new Queue(base_cmp_fifo, run_type);
                break;
            case 2:
                g_queue = new Queue(base_cmp_obj, run_type);
               // printf("Using objective-aware BFS");
                break;
            case 3:
                g_queue = new Queue(base_cmp_lb, run_type);
                break;
            case 4:
                g_queue = new Queue(base_cmp_random, run_type);
                break;
        }
    }

    if (map_type == 1) {
        strcat(run_type, " Prefix Map\n");
        PrefixPermutationMap* prefix_pmap = new PrefixPermutationMap;
        g_pmap = (PermutationMap*) prefix_pmap;
    } else if (map_type == 2) {
        strcat(run_type, " Captured Symmetry Map\n");
        CapturedPermutationMap* cap_pmap = new CapturedPermutationMap;
        g_pmap = (PermutationMap*) cap_pmap;
    } else {
        strcat(run_type, " No Permutation Map\n");
        NullPermutationMap* null_pmap = new NullPermutationMap;
        g_pmap = (PermutationMap*) null_pmap;
    }

    g_tree = new CacheTree(nsamples, nrules, c, rules, labels, meta, ablation, calculate_size, type);
    if (g_verbosity.count("progress"))
        printf("%s", run_type);
    bbound_begin(g_tree, g_queue);
    return 0;
}

int run_corels_loop(size_t max_num_nodes, double beta, int fairness, int mode, bool useUnfairnessLB,
                        double min_fairness_acceptable, int kBest, int restart, int initNBNodes, double geomReason) {
    // Check arguments
    if(mode < 1 || mode > 4) {
        printf("Error : mode should be in {1, 2, 3, 4}\n");
        exit(-1);
    }
    if(kBest < 0) {
        printf("Error : kBest should be > 0\n");
        exit(-1);
    }
    if(min_fairness_acceptable < 0 || min_fairness_acceptable > 1) {
        printf("Error : min_fairness_acceptable should be in [0,1]\n");
        exit(-1);
    }
    if(restart < 0 || restart > 2) {
        printf("Error : restart must be exactly one of {0,1,2}\n");
    }
    if(restart == 1){ // Perform geometric restart
        //printf("[WARNING] Geometric restart is a beta feature.\n", initNBNodes, geomReason);
        /* INITIAL ITERATION */
        if(currLimit == -1) {
            currLimit = initNBNodes;
            usedRestart = true;
            printf("Will perform geometric restarts from %d to %lu.\n", currLimit, max_num_nodes);
        }
        if((g_tree->num_nodes() < currLimit) && !g_queue->empty()) {
            bbound_loop(g_tree, g_queue, g_pmap, beta, fairness, Gmaj_vect, Gmin_vect, mode, useUnfairnessLB,
                            min_fairness_acceptable, kBest, forbidSensAttr, accUpperBound); 
            return 0;
        } else {
            if(first) { // Update best known solution
                best_tree = g_tree;
                best_queue = g_queue;
                //printf("(First) Best solution updated : %lf\n", best_tree->min_objective());
                first = false;
            } else {
                if(g_tree->min_objective() < best_tree->min_objective()) {
                    // delete former best tree & queue and update it
                  /*  Node* node;
                    while (!best_queue->empty()) {
                        node = best_queue->front();
                        best_queue->pop();
                        if (node->deleted()) {
                            best_tree->decrement_num_nodes();
                            logger->removeFromMemory(sizeof(*node), DataStruct::Tree);
                            delete node;
                        }
                  }*/
                    if(best_queue)
                        delete best_queue;
                    if(best_tree)
                        delete(best_tree);
                    best_tree = g_tree;
                    best_queue = g_queue;
                    //printf("Best solution updated : %lf\n", best_tree->min_objective());
                } else {
                 /*   Node* node;
                    while (!g_queue->empty()) {
                        node = g_queue->front();
                        g_queue->pop();
                        if (node->deleted()) {
                            g_tree->decrement_num_nodes();
                            logger->removeFromMemory(sizeof(*node), DataStruct::Tree);
                            delete node;
                        }
                    }*/
                    if(g_queue)
                        delete g_queue;
                    if(g_tree)
                        delete(g_tree);
                }
            }
            
            currLimit *= geomReason;
            if(currLimit > max_num_nodes) {
                return -1;
            } else {
                //printf("New limit = %d \n", currLimit);
            }
            // Clear data structures
            if(g_pmap)
                delete g_pmap;
            g_pmap = nullptr;
            
            // Init new data structures for next iteration
            g_init = timestamp();
            char run_type[BUFSZ];
            strcpy(run_type, "LEARNING RULE LIST via ");
            char const *type = "node";
            strcat(run_type, "BFS");
            g_queue = new Queue(base_cmp_random, run_type);
            if (map_type_arg == 1) {
                strcat(run_type, " Prefix Map\n");
                PrefixPermutationMap* prefix_pmap = new PrefixPermutationMap;
                g_pmap = (PermutationMap*) prefix_pmap;
            } else if (map_type_arg == 2) {
                strcat(run_type, " Captured Symmetry Map\n");
                CapturedPermutationMap* cap_pmap = new CapturedPermutationMap;
                g_pmap = (PermutationMap*) cap_pmap;
            } else {
                strcat(run_type, " No Permutation Map\n");
                NullPermutationMap* null_pmap = new NullPermutationMap;
                g_pmap = (PermutationMap*) null_pmap;
            }
            g_tree = new CacheTree(nsamplesG, nrulesG, Gc, Grules, Glabels, Gmeta, ablationG, calculate_sizeG, type);

            bbound_begin(g_tree, g_queue);
            return 0;
        }
    } else if(restart == 2) { // Perform luby restart
        /* INITIAL ITERATION */
        if(currLimit == -1) {
            printf("Will perform Luby restarts from %d to %lu!\n",initNBNodes,max_num_nodes);
            //Compute luby sequence
            // 1) Compute size
            int v = 1;
            while((v*2)*initNBNodes <= max_num_nodes) {
                v = 2*v;
            }
            // 2) Init array
            lubySeq = (int*) malloc(v*sizeof(int));
            lubySeqSize = v;
            // 3) Fill in array
            v = 1;
            lubySeq[0] = 1;
            int ind0 = 1;
            // Fill in the (juxtaposed) precedent terms
            while(((v*2)*2)*initNBNodes <= max_num_nodes) {
                v = v * 2;
                int p = 0;
                while(p < (v-1)){
                    lubySeq[ind0+p] = lubySeq[ind0+p-(v-1)];
                    p++;
                }
                ind0+=(v-1);
                lubySeq[ind0] = v;
                ind0++;
            }
            // Finally print the last number to terminate the sequence
            lubySeq[lubySeqSize-1]=lubySeqSize;
            // 4) Print Luby seq
            /*printf("--- Final Luby Sequence : ---\n");
            int * printer = lubySeq;
            int cnt = 0;
            while(printer != 0 && cnt<lubySeqSize) {
                printf("%d ",*printer);
                printer++;
                cnt++;
            }
            printf("\n----------------------------\n");*/
            // Initialize the limit
            indLuby = 0;
            currLimit = initNBNodes*lubySeq[indLuby];
            usedRestart = true;
            //printf("Initial limit = %d \n", currLimit);
        }
        if((g_tree->num_nodes() < currLimit) && !g_queue->empty()) {
            bbound_loop(g_tree, g_queue, g_pmap, beta, fairness, Gmaj_vect, Gmin_vect, mode, useUnfairnessLB,
                            min_fairness_acceptable, kBest, forbidSensAttr, accUpperBound); 
            return 0;
        } else {
            if(first) { // Update best known solution
                best_tree = g_tree;
                best_queue = g_queue;
                best_map = g_pmap;
                //printf("(First) Best solution updated : %lf\n", best_tree->min_objective());
                first = false;
            } else {
                if(g_tree->min_objective() < best_tree->min_objective()) {
                    // delete former best tree & queue and update it
                    /* Node* node;
                    while (!best_queue->empty()) {
                        node = best_queue->front();
                        best_queue->pop();
                        if (node->deleted()) {
                            best_tree->decrement_num_nodes();
                            logger->removeFromMemory(sizeof(*node), DataStruct::Tree);
                            delete node;
                        }
                    }*/
                    if(best_queue)
                        delete best_queue;
                    best_queue = nullptr;
                    if(best_tree)
                        delete(best_tree);
                    best_tree = nullptr;
                    if(best_map)
                        delete best_map;
                    best_map = nullptr;

                    best_tree = g_tree;
                    best_queue = g_queue;
                    best_map = g_pmap;
                    //printf("Best solution updated : %lf\n", best_tree->min_objective());
                } else {
                    /*Node* node;
                    while (!g_queue->empty()) {
                        node = g_queue->front();
                        g_queue->pop();
                        if (node->deleted()) {
                            g_tree->decrement_num_nodes();
                            logger->removeFromMemory(sizeof(*node), DataStruct::Tree);
                            delete node;
                        }
                    }*/
                    if(g_queue)
                        delete g_queue;
                    g_queue = nullptr;
                    if(g_tree)
                        delete(g_tree);
                    g_tree = nullptr;
                    if(g_pmap)
                        delete g_pmap;
                    g_pmap = nullptr;
                }
            }
            indLuby++;
            if(indLuby >= lubySeqSize) {
                return -1;
            } else {
                currLimit = initNBNodes*lubySeq[indLuby];
                //printf("New limit = %d \n", currLimit);
            }           
            // Init new data structures for next iteration
            g_init = timestamp();
            char run_type[BUFSZ];
            strcpy(run_type, "LEARNING RULE LIST via ");
            char const *type = "node";
            strcat(run_type, "BFS");
            g_queue = new Queue(base_cmp_random, run_type);
            if (map_type_arg == 1) {
                strcat(run_type, " Prefix Map\n");
                PrefixPermutationMap* prefix_pmap = new PrefixPermutationMap;
                g_pmap = (PermutationMap*) prefix_pmap;
            } else if (map_type_arg == 2) {
                strcat(run_type, " Captured Symmetry Map\n");
                CapturedPermutationMap* cap_pmap = new CapturedPermutationMap;
                g_pmap = (PermutationMap*) cap_pmap;
            } else {
                strcat(run_type, " No Permutation Map\n");
                NullPermutationMap* null_pmap = new NullPermutationMap;
                g_pmap = (PermutationMap*) null_pmap;
            }
            g_tree = new CacheTree(nsamplesG, nrulesG, Gc, Grules, Glabels, Gmeta, ablationG, calculate_sizeG, type);

            bbound_begin(g_tree, g_queue);
            return 0;
        }
    } 
    else { // Normal run (no restart)
            if((g_tree->num_nodes() < max_num_nodes) && !g_queue->empty() && (callsNB > 0)) {
                bbound_loop(g_tree, g_queue, g_pmap, beta, fairness, Gmaj_vect, Gmin_vect, mode, useUnfairnessLB,
                                min_fairness_acceptable, kBest, forbidSensAttr, accUpperBound);
                callsNB--;
                if(callsNB == 0){
                    if(debugRun){
                        printf("Performed max allowed #calls to bbound_loop (%d)\n", initCallsNB);
                    }
                }
                if(max_num_nodes <= g_tree->num_nodes()){
                     if(debugRun){
                        printf("Exiting because max #nodes in the trie was reached : %d/%d\n", max_num_nodes, g_tree->num_nodes());
                    }
                }
                return 0;
            }
    }
    return -1;
}

double run_corels_end(int** rulelist, int* rulelist_size, int** classes, double** confScores, int early, int latex_out, rule_t* rules, rule_t* labels, char* opt_fname, unsigned long** runStats)
{
    if(debugRun){
        printf("Performed %d calls to bbound_loop.\n", initCallsNB - callsNB);
    }
    if(usedRestart) {
        g_tree = best_tree;
        g_queue = best_queue;
        g_pmap = best_map;
    }
    std::vector<unsigned long> vals = bbound_end(g_tree, g_queue, g_pmap, early, Grules, Glabels);
    const tracking_vector<unsigned short, DataStruct::Tree>& r_list = g_tree->opt_rulelist();
    const tracking_vector<bool, DataStruct::Tree>& preds = g_tree->opt_predictions();
    const vector<double> scores = g_tree->getConfScores();
    *runStats = (unsigned long*)malloc(sizeof(unsigned long) * 2); // Confidence scores
    if(debugRun){
        printf("nb explored = %lu, nb cache = %lu\n", vals[0], vals[1]);
    }
    (*runStats)[0] = vals[0];
    (*runStats)[1] = vals[1];
    //double accuracy = 1.0 - g_tree->min_objective() + g_tree->c() * r_list.size();
    double accuracy = g_tree->getFinalAcc();
    *rulelist = (int*)malloc(sizeof(int) * r_list.size()); // Antecedents
    *classes = (int*)malloc(sizeof(int) * (1 + r_list.size())); // Consequents
    *confScores = (double*)malloc(sizeof(double) * (1 + r_list.size())); // Confidence scores
    *rulelist_size = r_list.size();
    for(size_t i = 0; i < r_list.size(); i++) {
        (*rulelist)[i] = r_list[i]; // Condition i
        (*confScores)[i] = scores[i]; // Confidence score for rule i
        (*classes)[i] = preds[i]; // Pred i
    }
    (*confScores)[r_list.size()] = (scores)[r_list.size()];
    (*classes)[r_list.size()] = preds.back(); // Default prediction
    if (g_verbosity.count("progress")) {
        printf("final num_nodes: %zu\n", g_tree->num_nodes());
        printf("final num_evaluated: %zu\n", g_tree->num_evaluated());
        printf("final min_objective: %1.5f\n", g_tree->min_objective());
        printf("final accuracy: %1.5f\n", accuracy);
        printf("final total time: %f\n", time_diff(g_init));
    }
    if(opt_fname) {
        print_final_rulelist(r_list, g_tree->opt_predictions(), latex_out, Grules, Glabels, opt_fname, g_tree->getConfScores());
        logger->dumpState();
        logger->closeFile();
    }
    if(g_tree)
        delete g_tree;
    g_tree = nullptr;
    if(g_pmap)
        delete g_pmap;
    g_pmap = nullptr;
    if(g_queue)
        delete g_queue;
    g_queue = nullptr;
    // Reset all variables' values
    first = true;
    SPolicy = -1;
    modeBFS = -1;
    map_type_arg = -1;
    currLimit = -1;
    usedRestart = false;
    nsamplesG = -1;
    nrulesG = -1;
    ablationG = -1;
    calculate_sizeG = -1;
    Gc = -1;
    // TODO : free vectors (?)
    if(lubySeqSize!= -1){
        free(lubySeq);
        lubySeqSize = -1;
        indLuby = -1;
    }  
    //printf("Final accuracy = %lf\n", accuracy);
    return accuracy;
}
