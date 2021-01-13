#pragma once

#include "pmap.hh"
#include "alloc.hh"
#include <functional>
#include <queue>
#include <set>
#include <cstdlib>
// pass custom allocator function to track memory allocations in the queue
typedef std::priority_queue<Node*, tracking_vector<Node*, DataStruct::Queue>, 
        std::function<bool(Node*, Node*)> > q;

// orders based on depth (BFS)
static std::function<bool(Node*, Node*)> base_cmp = [](Node* left, Node* right) {
    return left->depth() >= right->depth();
};

// Custom BFS policies
static std::function<bool(Node*, Node*)> base_cmp_fifo = [](Node* left, Node* right) {
    if(left->depth() == right->depth()) {
        return (left->get_num() >= right->get_num());
    } else {
        return left->depth() >= right->depth();
    }
};

static std::function<bool(Node*, Node*)> base_cmp_obj = [](Node* left, Node* right) {
    if(left->depth() == right->depth()) {
        return (left->objective() >= right->objective());
    } else {
        return left->depth() >= right->depth();
    }
};

static std::function<bool(Node*, Node*)> base_cmp_lb = [](Node* left, Node* right) {
    if(left->depth() == right->depth()) {
        return (left->lower_bound() >= right->lower_bound());
    } else {
        return left->depth() >= right->depth();
    }
};

static std::function<bool(Node*, Node*)> base_cmp_random = [](Node* left, Node* right) {
    if(left->depth() == right->depth()) {
        if((rand() % 2)) {
            return false;
        } else {
            return true;
        }
    } else {
        return left->depth() >= right->depth();
    }
};

// orders based on curiosity metric.
static std::function<bool(Node*, Node*)> curious_cmp = [](Node* left, Node* right) {
    return left->get_curiosity() >= right->get_curiosity();
};

// orders based on lower bound.
static std::function<bool(Node*, Node*)> lb_cmp = [](Node* left, Node* right) {
    return left->lower_bound() >= right->lower_bound();
};

// orders based on objective.
static std::function<bool(Node*, Node*)> objective_cmp = [](Node* left, Node* right) {
    return left->objective() >= right->objective();
};

// orders based on depth (DFS)
static std::function<bool(Node*, Node*)> dfs_cmp = [](Node* left, Node* right) {
    return left->depth() <= right->depth();
};

class Queue {
    public:
        Queue(std::function<bool(Node*, Node*)> cmp, char const *type);
        // by default, initialize this as a BFS queue
        Queue() : Queue(base_cmp, "BFS") {};
        ~Queue();
        Node* front() {
            return q_->top();
        }
        inline void pop() {
            q_->pop();
        }
        void push(Node* node) {
            q_->push(node);
        }
        size_t size() {
            return q_->size();
        }
        bool empty() {
            return q_->empty();
        }
        inline char const * type() {
            return type_;
        }

        std::pair<Node*, tracking_vector<unsigned short, DataStruct::Tree> > select(int kBest, CacheTree* tree, VECTOR captured) {
            int cnt;
            tracking_vector<unsigned short, DataStruct::Tree> prefix;
            Node *selected_node, *node;
            bool valid = true;
            double lb;
            int nb = 0;
            if(kBest > 1) {
                nb = (rand() % kBest);
            }
            do {
                selected_node = q_->top();
                q_->pop();
                if(q_->size() < nb) { // if less elements in queue than random number found
                    nb = q_->size(); // we take the last one of the queue
                }
                if(nb > 0) {
                    int ind = nb;
                    // k nodes + the first popped
                    Node ** kBest = (Node**) malloc((nb+1)*sizeof(Node*));
                    kBest[nb] = selected_node;
                    // extract the best element k times
                    while(nb > 0) {
                        nb--;
                        kBest[nb] = q_->top();
                        q_->pop();
                    }
                    // keep the kTh extracted as the best
                    selected_node = kBest[0];
                    // push back the k others
                    while(ind > 0) {
                        q_->push(kBest[ind]);
                        ind--;
                    }
                    free(kBest);
                }
                if (tree->ablation() != 2)
                    lb = selected_node->lower_bound() + tree->c();
                else
                    lb = selected_node->lower_bound();
                logger->setCurrentLowerBound(lb);

                node = selected_node;
                // delete leaf nodes that were lazily marked
                if (node->deleted() || (lb >= tree->min_objective())) {
                    tree->decrement_num_nodes();
                    logger->removeFromMemory(sizeof(*node), DataStruct::Tree);
                    delete node;
                    valid = false;
                } else {
                    valid = true;
                }
            } while (!q_->empty() && !valid); 
            if (!valid) {
                return std::make_pair((Node*)NULL, prefix);
            }

            rule_vclear(tree->nsamples(), captured);
            while (node != tree->root()) {
                rule_vor(captured,
                         captured, tree->rule(node->id()).truthtable,
                         tree->nsamples(), &cnt);
                prefix.push_back(node->id());
                node = node->parent();
            }
            std::reverse(prefix.begin(), prefix.end());
            return std::make_pair(selected_node, prefix);
        }

    private:
        q* q_;
        char const *type_;
        int kBest_;
};



extern void bbound_begin(CacheTree* tree, Queue* q);

extern std::vector<unsigned long> bbound_end(CacheTree* tree, Queue* q, PermutationMap* p, bool early, rule_t* rules, rule_t* labels);

// datastructures
struct confusion_matrix {
    int nTP;
    int nFP;
    int nFN;
    int nTN;
    double nPPV;
    double nTPR;
    double nFDR;
    double nFPR;
    double nFOR;
    double nFNR;
    double nNPV;
    double nTNR;
    double min_ppv;
    double max_ppv;
    double min_fpr;
    double max_fpr;
    double min_fnr;
    double max_fnr;
    int min_tp;
    int max_tp;
    int min_fp;
    int max_fp;
    int min_tn;
    int max_tn;
    int min_fn;
    int max_fn;
};

struct solver_args {
    int nb_sp_plus;
    int nb_sp_minus;
    int nb_su_plus;
    int nb_su_minus;
    int L;
    int U;
    float fairness_tolerence; 
    int TPp;
    int FPp;
    int TNp; 
    int FNp;
    int TPu;
    int FPu;
    int TNu;
    int FNu;
};

struct confusion_matrix_groups {
    confusion_matrix minority;
    confusion_matrix majority;
    double unfairnessLB;
    double predparityLB;
    double predequalityLB;
    double equalOppLB;
};

struct fairness_metrics {
    double statistical_parity;
    double predictive_parity;
    double predictive_equality;
    double equal_opportunity;
    double equalized_odds;
    double cond_use_acc_equality;
};


// adding extra parameters for fairness

extern void bbound_loop(CacheTree* tree, 
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
                            double accuracyUpperBound);

extern void evaluate_children(CacheTree* tree, 
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
                                double accuracyUpperBound);



// confusion matrix and fairness metrics


extern confusion_matrix_groups compute_confusion_matrix(VECTOR parent_prefix_predictions,
                                                CacheTree* tree,
                                                VECTOR parent_not_captured, 
                                                VECTOR captured, 
                                                rule_t* maj_v,
                                                rule_t* min_v,
                                                int prediction, 
                                                int default_prediction);


extern fairness_metrics compute_fairness_metrics(confusion_matrix_groups cmg);



