# Faircorels

Welcome to FairCorels, a Python library for learning fair and interpretable models.
The use of Python 3 is strongly recommended !
Feel free to point out any issue you may encounter while using our package, or to recommend new features!

Email contact: <jferry@laas.fr>

Note that we released a new version of this module, named [FairCORELSV2](https://github.com/ferryjul/fairCORELSV2), integrating advanced pruning techniques to efficiently explore the search space of fair rule lists.

## References

This repository contains the implementation of the method introduced in the paper [Learning fair rule lists](https://arxiv.org/abs/1909.03977):
```
[1] Ulrich Aïvodji, Julien Ferry, Sébastien Gambs, Marie-José Huguet, and Mohamed Siala. 2019. "Learning fair rule lists." arXiv preprint arXiv:1909.03977.
```

We presented our package in a Demo paper [FairCORELS, an Open-Source Library for Learning Fair Rule Lists](https://dl.acm.org/doi/abs/10.1145/3459637.3481965) at the 30th ACM International Conference on Information & Knowledge Management (CIKM'21):
```
[2] Ulrich Aïvodji, Julien Ferry, Sébastien Gambs, Marie-José Huguet, and Mohamed Siala. 2021. FairCORELS, an Open-Source Library for Learning Fair Rule Lists. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (CIKM '21). Association for Computing Machinery, New York, NY, USA, 4665–4669. DOI:https://doi.org/10.1145/3459637.3481965
```

## Overview

FairCORELS is a bi-objective extension of the CORELS algorithm, handling jointly accuracy and fairness.
The main classifier object included in our module is `FairCorelsClassifier`.  It implements the fairCORELS method. 
The currently supported fairness notions are : statistical parity, predictive parity, predictive equality, equal opportunity, equalized odds, and conditional use accuracy equality (see Table 1 of [2] for details of the computations).
However, the core algorithm is metric-agnostic and any function of the confusion matrix of a classifier could be integrated.

Our module also includes a wrapper for ensemble learning: `FairCorelsBagging`. 
The `FairCorelsBagging` class provides a wrapper to perform the Bagging ensemble method using `FairCorelsClassifier` as a base learner.
Note that `FairCorelsBagging` is not maintained.

## Examples

### Basic example

```python
from faircorels import *

# Load the dataset
X, y, features, prediction = load_from_csv("data/compas_rules_full.csv")

# Define protected and unprotected groups
# Here, we want them to correspond to features 0 and 1 (which we display)
# However, they can be any binary vector
sensitive_attr_column = 0
unsensitive_attr_column = 1
print("Sensitive attribute is ", features[sensitive_attr_column])
print("Unsensitive attribute is ", features[unsensitive_attr_column])
sensVect =  X[:,sensitive_attr_column]
unSensVect =  X[:,unsensitive_attr_column] 

# Define the desired fairness level (which is exactly (1.0 minus the unfairness tolerance) - see Table 1 of [2] for details of the computation)
epsilon = 0.98 # max. unfairness tolerance of 0.02 (fairness level of 98%)

# Create the model, with 10000 as the maximum number of iterations 
c = FairCorelsClassifier(n_iter=1000000, # maximum number of nodes in the prefix tree
                        c=0.001, # regularization parameter for sparsity
                        max_card=1, # each antecedent will have cardinality one (recommended if rule mining is done as preprocessing)
                        min_support = 0.01, # each rule antecedent must capture at least 1% of the training instances
                        policy="bfs", # exploration heuristic
                        bfs_mode=2, # exploration heuristic
                        mode=3, # epsilon-constrained mode
                        fairness=1, # statistical fairness metric to be used, 1 stands for statistical parity
                        epsilon=epsilon,  # epsilon is the unfairness tolerance
                        maj_vect=unSensVect, # unSensVect is a binary vector indicating unprotected group membership for all examples of X
                        min_vect=sensVect # unSensVect is a binary vector indicating protected group membership for all examples of X
                        )

# Fit the classifier
c.fit(X, y, features=features, prediction_name=prediction)

# Score the model on the training set
a = c.score(X, y)

# Compute its unfairness
cm = ConfusionMatrix(sensVect, unSensVect, c.predict(X), y)
cm_minority, cm_majority = cm.get_matrix()
fm = Metric(cm_minority, cm_majority)
unf = fm.statistical_parity()

# Print the model's performances
print("Training accuracy = %f, training unfairness = %f" %(a, unf))

# Print the model itself
print(c.rl_)
```

### Complete examples

A step-by-step example notebook `Demo-fairCORELS.ipynb` can be found under the `example` folder.

Detailed example files, using 5-folds cross-validation for the COMPAS dataset, are also provided in the `example` directory :

* `example-compas.py` uses the `FairCorelsClassifier` classifier

* `example-bagging-compas.py` uses the `FairCorelsBagging` classifier

All files show how to load data, how to train our classifiers, how to evaluate them, and how to store results in a clear and easily exploitable manner.


## Installation
#### Ubuntu

```
sudo apt install libgmp-dev
pip install faircorels
```
Note that running the provided example scripts after installing `faircorels` might raise errors about Numpy versions. In this case, simply uninstall the lastly installed Numpy (`pip uninstall numpy`) and the issue should be solved.

#### Mac

```
# Install g++ and gmp
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install g++ gmp

pip install faircorels
```

#### Windows
Note: Python 2 is currently NOT supported on Windows.
```
pip install faircorels
```


## Detail of the classifiers' parameters :

### FairCorelsClassifier :

#### Constructor arguments :

* `c` : float, optional (default=0.01)
        Regularization parameter. Higher values penalize longer rulelists.

* `n_iter` : int, optional (default=1000)
        Maximum number of nodes (rulelists) to search before exiting.

* `map_type` : str, optional (default="prefix")
    The type of prefix map to use. Supported maps are "none" for no map,
    "prefix" for a map that uses rule prefixes for keys, "captured" for
    a map with a prefix's captured vector as keys.

* `policy` : str, optional (default="lower_bound")
    The search policy for traversing the tree (i.e. the criterion with which
    to order nodes in the queue). Supported criteria are "bfs", for breadth-first
    search; "curious", which attempts to find the most promising node; 
    "lower_bound" which is the objective function evaluated with that rulelist
    minus the default prediction error; "objective" for the objective function
    evaluated at that rulelist; and "dfs" for depth-first search.

* `verbosity` : list, optional (default=["rulelist"])
    The verbosity levels required. A list of strings, it can contain any
    subset of ["rulelist", "rule", "label", "minor", "samples", "progress", "mine", "loud"].

    - "rulelist" prints the generated rulelist at the end.
    - "rule" prints a summary of each rule generated.
    - "label" prints a summary of the class labels.
    - "minor" prints a summary of the minority bound.
    - "samples" produces a complete dump of the rules, label, and/or minor data. You must also provide at least one of "rule", "label", or "minor" to specify which data you want to dump, or "loud" for all data. The "samples" option often spits out a lot of output.
    - "progress" prints periodic messages as faircorels runs.
    - "mine" prints debug information while mining rules, including each rule as it is generated.
    - "loud" is the equivalent of ["progress", "label", "rule", "mine", "minor"].

* `ablation` : int, optional (default=0)
    Specifies addition parameters for the bounds used while searching. Accepted
    values are 0 (all bounds), 1 (no antecedent support bound), and 2 (no
    lookahead bound).

* `max_card` : int, optional (default=2)
    Maximum cardinality allowed when mining rules. Can be any value greater than
    or equal to 1. For instance, a value of 2 would only allow rules that combine
    at most two features in their antecedents.

* `min_support` : float, optional (default=0.01)
    The fraction of samples that a rule must capture in order to be used. 1 minus
    this value is also the maximum fraction of samples a rule can capture.
    Can be any value between 0.0 and 0.5.

* `beta` : float, optional (default=0.0)
    The weight the unfairness in the objective function

* `fairness` : int optional (default=1)
    The type of fairness metric used. 
    1 : statistical parity, 2 : predictive parity, 3 : predictive equality, 4 : equal opportunity, 5 : equalized odds, 6 : conditional use accuracy equality

* `maj_pos` : int optional (default=-1)
    The position of the rule that defined the majority group
    If not specified, all individuals not in minority group are in majority group
    Not used if maj_vect is used

* `min_pos` : int optional (default=2)
    The position of the rule that defined the minority group
    Not used if min_vect is used

* `maj_vect` : int list optional (default=[-1])
    List of integers in {0,1} indicating if instances belong to the majority group
    If not specified, this vector is computed using maj_pos

* ` min_vect` : int list optional (default=[-1])
    List of integers in {0,1} indicating if instances belong to the minority group
    If not specified, this vector is computed using min_pos

* `mode` : int optional (default=3)
    Method used for the multi-ojective framework
    1: weigted sum, 2: maximum fairness, 3: epsilon-constraint, 4: maximum accuracy

* `useUnfairnessLB` : bool optional (default=False)
    Use the unfairness lower bound
    -> For some metrics (SP and EO), an improved filtering using CP is applied
    For some others, it is not implemented and a simple, less effective lower bound is computed and used

* `epsilon` : float optional (default=0.05)
    max acceptable unfairness

* `kbest` : int optional (default=1)
    Randomly use one of the k best objectives

* `forbidSensAttr` : bool optional (default=False)
    Forbid the use of the sensitive and (if specified) unsensitive attributes in the produced rule lists
    WARNING : This attribute is not used anymore in the current implementation. Remove columns manually from training data.

* `bfs_mode` : int optinal (default=0)
    BFS strategy
    0: original CORELS, 1:FIFO, 2:objective_aware, 3:lower_bound, 4:random

* `random_state` : int optional (default=42)
    Random seed for randomized search

#### Methods :

##### .fit(X, y, features=[], prediction_name="prediction", performRestarts=0, initNBNodes=1000, geomRReason=1.5, max_evals=1000000000, time_limit = None):

Method for training the classifier.

* `X` : array-like, shape = [n_samples, n_features]
    The training input samples. All features must be binary, and the matrix is internally converted to dtype=np.uint8.

* `y` : array-line, shape = [n_samples]
    The target values for the training input. Must be binary.
        
* `features` : list, optional(default=[])
    A list of strings of length n_features. Specifies the names of each of the features. If an empty list is provided, the feature names are set to the default of ["feature1", "feature2"... ].

* `prediction_name` : string, optional(default="prediction")
    The name of the feature that is being predicted.

* `max_evals` : int, maximum number of calls to evaluate_children 
    (ie maximum number of nodes explored in the prefix tree)

* `time_limit` : int, maximum number of seconds allowed for the model
    building
    Note that this specifies the CPU time and NOT THE WALL-CLOCK TIME

* `performRestarts` : int optional (default=0)
    0 : Does not perform any restarts,
    1 : Performs geometric restarts,
    2 : Performs Luby restarts
    WARNING : Note that restarts are an old-feature, not maintained anymore.

* `initNBNodes` : int optional (default=1000)
    Initial limit for the number of nodes in the trie when performing restarts
    WARNING : Note that restarts are an old-feature, not maintained anymore.

* `geomRReason` : double optional (default=1.5)
    When performRestarts=True, geomRReason is the reason used 
    for the geometric restart calculation
    WARNING : Note that restarts are an old-feature, not maintained anymore.

##### .predict(X):
Method for predicting using the trained classifier.

* `X` : array-like, shape = [n_samples, n_features]
    The training input samples. All features must be binary, and the matrix is internally converted to dtype=np.uint8. The features must be the same as those of the data used to train the model.

=> Returns : `p` : array of shape = [n_samples] -> The classifications of the input samples.

##### .predict_with_scores(X):
Method for predicting using the trained classifier.

* `X` : array-like, shape = [n_samples, n_features]
    The training input samples. All features must be binary, and the matrix is internally converted to dtype=np.uint8. The features must be the same as those of the data used to train the model.

=> Returns : `p` : array of shape = [[n_samples],[n_samples]].
    The first array contains the classifications of the input samples.
    The second array contains the associated confidence scores.

##### .score(X, y):
Method that scores the algorithm on the input samples X with the labels y. Alternatively, score the predictions X against the labels y (where X has been generated by `predict` or something similar).

* `X` : array-like, shape = [n_samples, n_features] OR shape = [n_samples]
    The input samples, or the sample predictions. All features must be binary.
        
* `y` : array-like, shape = [n_samples]
    The input labels. All labels must be binary.

=> Returns : `a` : float
    The accuracy, from 0.0 to 1.0, of the rulelist predictions

##### .get_params():
Method to get a list of all the model's parameters.

=> Returns : `params` : dict
Dictionary of all parameters, with the names of the parameters as the keys

##### .set_params(params):
Method to set some of the model's parameters.

* `params` :  Set of model parameters. Takes an arbitrary number of      
keyword parameters, all of which must be valid parameter names (i.e. must be included in those returned by get_params).

##### .save(fname):
Method to save the model to a file, using python's pickle module.

* `fname` : string
    File name to store the model in

##### .load(fname):
Method to load a model from a file, using python's pickle module.

* `fname` : string
    File name to load the model from
        
##### .rl(set_val=None):
Method to return or set the learned rulelist
        
* `set_val` : RuleList, optional
    Rulelist to set the model to

=> Returns : `rl` : obj
    The model's rulelist

##### .__str__():
Method to get a string representation of the rule list

=> Returns : `rl` : str
    The rule list

##### .__repr__():
Same behavior as the previous one.

##### .explain(anEx):
Method to explain a prediction (by providing the matching rule).

* `anEx` : array-like, shape = [n_features] 
    The input sample

=> Returns : list `l` where
    `l[0]` is the instance's prediction
    `l[1]` is the implicant(s) that led to that decision
    (both are strings - user friendly)

##### .explain_api(anEx):
Method to explain a prediction (by providing the matching rule) (shorter output).

* `anEx` : array-like, shape = [n_features] 
    The input sample

=> Returns : list `l` where
    `l[0]` is the instance's prediction
    `l[1]` is the implicant(s) that led to that decision
    (both are API-oriented - easy to use by a program)
   
##### .explain_long(anEx):
Method to explain a prediction (by providing the matching rule and all the previous unmatched implicants).

* `anEx` : array-like, shape = [n_features] 
    The input sample

=> Returns : list `l` where
    `l[0]` is the instance's prediction
    `l[1]` is the implicant(s) that led to that decision
    (both are strings - user friendly)

##### .explain_long_api(anEx):
Method to explain a prediction (by providing the matching rule and all the previous unmatched implicants) (shorter output).

* `anEx` : array-like, shape = [n_features] 
    The input sample

=> Returns : list `l` where
    `l[0]` is the instance's prediction
    `l[1]` is the implicant(s) that led to that decision
    (both are API-oriented - easy to use by a program)

### FairCorelsBagging :

This class provides an easy wrapper for the Bagging method that uses the `FairCorelsClassifier` class as underlying base learning. Hence, arguments directly passed to the `FairCorelsClassifier` object will not be detailed again.

#### Constructor arguments :
The training sets for the different base learners are automatically computed from the entire provided training set, using the provided parameters.

* `n_learners` : int, number of base learners of type `FairCorelsClassifier` that will be used

* `sample_size` : size of one subsample (used to train exactly one base learner)

* Note that, as training sets will be automatically determined here, the training set `X`, `y`, the features `features`, the textual prediction name `prediction_name` must be provided directly here (NOT IN THE .fit call)

* Other `FairCorelsClassifier` arguments for the base learners : `c=0.01`, `n_iter=10000`, `map_type="prefix"`, `policy="lower_bound"`, `verbosity=["rulelist"]`, `ablation=0`, `max_card=2`, `min_support=0.01`, `beta=0.0`, `fairness=1`, `maj_pos=-1`, `min_pos=2`, `maj_vect = np.empty(shape=(0))`, `min_vect = np.empty(shape=(0))`, `mode=4`, `useUnfairnessLB=False`, `epsilon=0.0`, `kbest=1`, `forbidSensAttr=False`, `bfs_mode=0`, `random_state=42` and `baggingVerbose=0`
  
#### Methods :

##### .fit(performRestarts=0, initNBNodes=1000, geomRReason=1.5, max_evals=1000000000, time_limit = None, n_workers=-1):
Method to train the base learners.

* `n_workers` : int, maximum number of threads allowed to parallelize the training of the different learners

* + all `FairCorelsClassifier` attributes used to create the base learners

##### .predict(X):
Predict classifications of the input samples X. Uses majority vote as aggregation function.

* `X` : array-like, shape = [n_samples, n_features]
    The training input samples. All features must be binary, and the matrix is internally converted to dtype=np.uint8. The features must be the same as those of the data used to train the model.

=> Returns : `p` : array of shape = [n_samples].
    The classifications of the input samples.

##### .score(X, y):
Score the algorithm on the input samples X with the labels y. Alternatively, score the predictions X against the labels y (where X has been generated by `predict` or something similar).

* `X` : array-like, shape = [n_samples, n_features] OR shape = [n_samples]
    The input samples, or the sample predictions. All features must be binary.
    
* `y` : array-like, shape = [n_samples]
    The input labels. All labels must be binary.

=> Returns :  `a` : float
    The accuracy, from 0.0 to 1.0, of the rulelist predictions

##### .explain(anInst):
Explains a prediction (by its matching rules among majority base learners).

* `anInst` : array-like, shape = [n_features]
    The input sample

=> Returns : `d` : dictionnary `{'implicants':i, 'prediction':p}`
    where `i` is the list of implicants that led majority voters to their prediction and `p` is the associated prediction

##### .explain_complete(anInst):
Explains a prediction (adds complete implications, including antecedents negation for all learners).

* `anInst` : array-like, shape = [n_features]
    The input sample

=> Returns : `d` : dictionnary `{'implicants':i, 'prediction':p}`
    where `i` is the list of implicants (and unmatched rules) that led majority voters to their prediction and `p` is the associated prediction
