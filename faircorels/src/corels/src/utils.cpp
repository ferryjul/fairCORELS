#include "utils.hh"
#include <stdio.h>
#include <assert.h>
#include <sstream>

std::string sizet_tostring(size_t v) {
    std::ostringstream ss;
    ss << v;
    return ss.str();
}

Logger::Logger(double c, size_t nrules, int verbosity, char* log_fname, int freq) {
      _c = c;
      _nrules = nrules - 1;
      _v = verbosity;
      _freq = freq;
      setLogFileName(log_fname);
      initPrefixVec();
}

/*
 * Sets the logger file name and writes the header line to the file.
 */
void Logger::setLogFileName(char *fname) {
    if (_v < 1) return;

    printf("writing logs to: %s\n\n", fname);
    _f.open(fname, ios::out | ios::trunc);

    _f << "total_time,evaluate_children_time,node_select_time,"
       << "rule_evaluation_time,lower_bound_time,lower_bound_num,"
       << "objective_time,objective_num,"
       << "tree_insertion_time,tree_insertion_num,queue_insertion_time,evaluate_children_num,"
       << "permutation_map_insertion_time,permutation_map_insertion_num,permutation_map_memory,"
       << "current_lower_bound,tree_min_objective,tree_prefix_length,"
       << "tree_num_nodes,tree_num_evaluated,tree_memory,"
       << "queue_size,queue_min_length,queue_memory,"
       << "pmap_size,pmap_null_num,pmap_discard_num,"
       << "log_remaining_space_size,prefix_lengths" << endl;
}

/*
 * Writes current stats about the execution to the log file.
 */
void Logger::dumpState() {
    // update timestamp here
    setTotalTime(time_diff(_state.initial_time));

    size_t space_size = 0;

#ifdef GMP
    space_size = getLogRemainingSpaceSize();
#endif

    _f << _state.total_time << ","
       << _state.evaluate_children_time << ","
       << _state.node_select_time << ","
       << _state.rule_evaluation_time << ","
       << _state.lower_bound_time << ","
       << _state.lower_bound_num << ","
       << _state.objective_time << ","
       << _state.objective_num << ","
       << _state.tree_insertion_time << ","
       << _state.tree_insertion_num << ","
       << _state.queue_insertion_time << ","
       << _state.evaluate_children_num << ","
       << _state.permutation_map_insertion_time << ","
       << _state.permutation_map_insertion_num << ","
       << _state.pmap_memory << ","
       << _state.current_lower_bound << ","
       << _state.tree_min_objective << ","
       << _state.tree_prefix_length << ","
       << _state.tree_num_nodes << ","
       << _state.tree_num_evaluated << ","
       << _state.tree_memory << ","
       << _state.queue_size << ","
       << _state.queue_min_length << ","
       << _state.queue_memory << ","
       << _state.pmap_size << ","
       << _state.pmap_null_num << ","
       << _state.pmap_discard_num << ","
       << space_size << ","
       << dumpPrefixLens().c_str() << endl;
}

#ifdef GMP
/*
 * Uses GMP library to dump a string version of the remaining state space size.
 * This number is typically very large (e.g. 10^20) which is why we use GMP instead of a long.
 * Note: this function may not work on some Linux machines.
 */
std::string Logger::dumpRemainingSpaceSize() {
    char* str = mpz_get_str(NULL, 10, _state.remaining_space_size);
    std::string ret(str);
    free(str);
    return ret;
}
#endif

/*
 * Function to convert vector of remaining prefix lengths to a string format for logging.
 */
std::string Logger::dumpPrefixLens() {
    std::string s = "";
    for(size_t i = 0; i < _nrules; ++i) {
        if (_state.prefix_lens[i] > 0) {
            s += sizet_tostring(i);
            s += ":";
            s += sizet_tostring(_state.prefix_lens[i]);
            s += ";";
        }
    }
    return s;
}

// Custom getline for Windows support
signed long long m_getline(char** lineptr, size_t* n, FILE* stream)
{
    if(lineptr == NULL || n == NULL || *lineptr != NULL || *n != 0)
        return -1;

    size_t block_size = 1024;
    size_t nblocks = 1;

    char* line = (char*)malloc(nblocks * block_size + 1);
    if(!line)
        return -1;
    line[0] = '\0';

    size_t total_size = 0;

    int c;
    while((c = fgetc(stream)) != EOF) {
        if(++total_size > nblocks * block_size) {
            line = (char*)realloc(line, ++nblocks * block_size + 1);
            if(!line)
                return -1;
        }

        line[total_size - 1] = c;
        if(c == '\n' || c == '\r') {
            break;
        }
    }
    line[total_size] = '\0';

    *n = nblocks * block_size + 1;
    *lineptr = line;

    if(total_size)
        return total_size;
    else
        return -1;
}

// Custom strsep for Windows support
char* m_strsep(char** stringp, char delim)
{
    if(stringp == NULL) {
        return NULL;
    }

    char* str = *stringp;
    if(str == NULL || *str == '\0') {
        return NULL;
    }

    char* out = NULL;

    while(1) {
        if(*str == delim || *str == '\0') {
            out = *stringp;
	        *stringp = (*str == '\0') ? NULL : str + 1;
            *str = '\0';
            break;
	    }
	    str++;
    }

    return out;
}

// Custom strdup for Windows support
char* m_strdup(char* str)
{
    if(!str)
        return NULL;

    size_t len = strlen(str);
    char* buf = (char*)malloc(len + 1);
    if(!buf)
        return NULL;

    memcpy(buf, str, len + 1);
    return buf;
}

compData computeFinalFairness(int nsamples,
                          const tracking_vector<unsigned short, DataStruct::Tree>& rulelist,
                          const tracking_vector<bool, DataStruct::Tree>& preds,                  
                          rule_t rules[],
                          rule_t labels[]) {
    compData result;
    // 1) We build the predictions' matrix
    VECTOR captured_it;
    VECTOR not_captured_yet;
    VECTOR captured_zeros;
    VECTOR preds_prefix;
    int nb;
    int nb2;
    int pm;
    vector<double> comp_scores(1+rulelist.size()); //double comp_scores[1+rulelist.size()];
    rule_vinit(nsamples, &captured_it);
    rule_vinit(nsamples, &not_captured_yet);
    rule_vinit(nsamples, &preds_prefix);
    rule_vinit(nsamples, &captured_zeros);
    // Initially not_captured_yet is full of ones
    rule_vor(not_captured_yet, labels[0].truthtable, labels[1].truthtable, nsamples ,&nb);
    // Initially preds_prefix is full of zeros
    rule_vand(preds_prefix, labels[0].truthtable, labels[1].truthtable, nsamples, &nb);
    tracking_vector<unsigned short, DataStruct::Tree>::iterator it;
    int totCaptured = 0;
    for (size_t i = 0; i < rulelist.size(); ++i) {
        rule_vand(captured_it, not_captured_yet, rules[rulelist[i]].truthtable, nsamples, &nb);
        rule_vandnot(not_captured_yet, not_captured_yet, captured_it, nsamples, &pm);
        rule_vand(captured_zeros, captured_it, labels[0].truthtable, nsamples, &nb2);
        totCaptured += nb;
        if(nb2 <= (nb - nb2)) { //then prediction is 1
            rule_vor(preds_prefix, preds_prefix, captured_it, nsamples, &nb);
            comp_scores[i] = (double) (nb - nb2) / (double) nb;
        } else {
            comp_scores[i] = (double) (nb2) / (double) nb;
        }
    }
    // number of zeros labeled instances falling into default decision
    rule_vand(captured_zeros, not_captured_yet, labels[0].truthtable, nsamples, &nb2);
    if(preds.back() == 1) { // else it is already OK
        rule_vor(preds_prefix, preds_prefix, not_captured_yet, nsamples, &pm);
        comp_scores[rulelist.size()] = (double) ((double) ((nsamples - totCaptured) - nb2)) /(double) (nsamples - totCaptured);
    }
    else {
        comp_scores[rulelist.size()] = (double) ((double) (nb2)) /(double) (nsamples - totCaptured);
    }
    // true positives, false negatives, true negatives, and false positives tables (for this rule)
    VECTOR A, D; // B, C;
    int tp, tn; //fp, fn;
    rule_vinit(nsamples, &A);
    //rule_vinit(nsamples, &B);
    rule_vinit(nsamples, &D);
    //rule_vinit(nsamples, &C);

    rule_vand(A, preds_prefix, labels[1].truthtable, nsamples, &tp);
    //rule_vandnot(B, labels[1].truthtable, preds_prefix, nsamples, &fn);
    rule_vandnot(D, labels[0].truthtable, preds_prefix, nsamples, &tn);
    //rule_vand(C, preds_prefix, labels[0].truthtable, nsamples, &fp);

    double acc = (double) (tp + tn)/nsamples;
    // 3) We compute group-specific metrics for the model

    // Free allocated VECTORS
    rule_vfree(&captured_it);
    rule_vfree(&not_captured_yet);
    rule_vfree(&captured_zeros);
    rule_vfree(&preds_prefix);
    rule_vfree(&A);
    //rule_vfree(&B);
    rule_vfree(&D);
   //rule_vfree(&C);
    result.conf_scores = comp_scores;
    result.accuracy = acc;
    return result;
}

/*
 * Given a rulelist and predictions, will output a human-interpretable form to a file.
 */
void print_final_rulelist(const tracking_vector<unsigned short, DataStruct::Tree>& rulelist,
                          const tracking_vector<bool, DataStruct::Tree>& preds,
                          const bool latex_out,
                          const rule_t rules[],
                          const rule_t labels[],
                          char fname[],
                          vector<double> confScores) {
    assert(rulelist.size() == preds.size() - 1);

    printf("\nOPTIMAL RULE LIST\n");
    if (rulelist.size() > 0) {
        printf("if (%s) then (%s) (conf %lf)\n", rules[rulelist[0]].features,
               labels[preds[0]].features, confScores[0]);
        for (size_t i = 1; i < rulelist.size(); ++i) {
            printf("else if (%s) then (%s) (conf %lf)\n", rules[rulelist[i]].features,
                   labels[preds[i]].features, confScores[i]);
        }
        printf("else (%s) (conf %lf)\n\n", labels[preds.back()].features, confScores[rulelist.size()]);

        if (latex_out) {
            printf("\nLATEX form of OPTIMAL RULE LIST\n");
            printf("\\begin{algorithmic}\n");
            printf("\\normalsize\n");
            printf("\\State\\bif (%s) \\bthen (%s)\n", rules[rulelist[0]].features,
                   labels[preds[0]].features);
            for (size_t i = 1; i < rulelist.size(); ++i) {
                printf("\\State\\belif (%s) \\bthen (%s)\n", rules[rulelist[i]].features,
                       labels[preds[i]].features);
            }
            printf("\\State\\belse (%s)\n", labels[preds.back()].features);
            printf("\\end{algorithmic}\n\n");
        }
    } else {
        printf("if (1) then (%s) (conf %lf)\n\n", labels[preds.back()].features, confScores[0]);

        if (latex_out) {
            printf("\nLATEX form of OPTIMAL RULE LIST\n");
            printf("\\begin{algorithmic}\n");
            printf("\\normalsize\n");
            printf("\\State\\bif (1) \\bthen (%s)\n", labels[preds.back()].features);
            printf("\\end{algorithmic}\n\n");
        }
    }

    ofstream f;
    printf("writing optimal rule list to: %s\n\n", fname);
    f.open(fname, ios::out | ios::trunc);
    for(size_t i = 0; i < rulelist.size(); ++i) {
        f << rules[rulelist[i]].features << "~"
          << preds[i] << ";";
    }
    f << "default~" << preds.back();
    f.close();
}
