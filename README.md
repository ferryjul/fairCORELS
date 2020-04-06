# Faircorels

Welcome to the python binding of the FairCORELS algorithm!
This repository contains the implementation of the method introduced in the paper [Learning fair rule lists!](https://arxiv.org/abs/1909.03977).

## Overview

FairCORELS is a modified version of CORELS to learn fair rule list. 


## Installation
#### Ubuntu

```
sudo apt install libgmp-dev
pip install faircorels
```

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


## Examples

### Large dataset, loaded from [this file](https://raw.githubusercontent.com/fingoldin/pycorels/master/examples/data/compas.csv)

```python
from faircorels import *

# Load the dataset
X, y = load_from_csv("data/compas.csv")

# Create the model, with 10000 as the maximum number of iterations 
c = CorelsClassifier(n_iter=10000)

# Fit, and score the model on the training set
a = c.fit(X, y).score(X, y)

# Print the model's accuracy on the training set
print(a)
```

A complete example file, using 5-fold cross-validation with the Adult Income dataset, is included : `examples/newExample.py`

## Detail of CorelsClassifier arguments :

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
    - "progress" prints periodic messages as corels runs.
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
    1 : statistical parity, 2 : predictive parity, 3 : predictive equality, 4 : equal opportunity

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

* `epsilon` : float optional (default=0.05)
    max acceptable unfairness

* `kbest` : int optional (default=1)
    Randomly use one of the k best objectives

* `forbidSensAttr` : bool optional (default=False)
    Forbid the use of the sensitive and (if specified) unsensitive attributes in the produced rule lists

* `bfs_mode` : int optinal (default=0)
    BFS strategy
    0: original CORELS, 1:FIFO, 2:objective_aware, 3:lower_bound, 4:random

* `random_state` : int optional (default=42)
    Random seed for randomized search

## Arguments for .fit :

* `performRestarts` : int optional (default=0)
    0 : Does not perform any restarts
    1 : Performs geometric restarts
    2 : Performs Luby restarts

* `initNBNodes` : int optional (default=1000)
    Initial limit for the number of nodes in the trie when perfomring

* `geomRReason` : double optional (default=1.5)
    When performRestarts=True, geomRReason is the reason used 
    for the geometric restart calculation
