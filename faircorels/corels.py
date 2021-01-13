from __future__ import print_function, division, with_statement
from ._corels import fit_wrap_begin, fit_wrap_end, fit_wrap_loop, predict_wrap, predict_score_wrap
from .utils import check_consistent_length, check_array, check_is_fitted, get_feature, check_in, check_features, check_rulelist, RuleList, computeAccuracyUpperBound
import numpy as np
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm # tentative display
from metrics import ConfusionMatrix, Metric

debug = False

class CorelsClassifier:
    """Certifiably Optimal RulE ListS classifier.

    This class implements the CORELS algorithm, designed to produce human-interpretable, optimal
    rulelists for binary feature data and binary classification. As an alternative to other
    tree based algorithms such as CART, CORELS provides a certificate of optimality for its 
    rulelist given a training set, leveraging multiple algorithmic bounds to do so.

    In order to use run the algorithm, create an instance of the `CorelsClassifier` class, 
    providing any necessary parameters in its constructor, and then call `fit` to generate
    a rulelist. `printrl` prints the generated rulelist, while `predict` provides
    classification predictions for a separate test dataset with the same features. To determine 
    the algorithm's accuracy, run `score` on an evaluation dataset with labels.
    To save a generated rulelist to a file, call `save`. To load it back from the file, call `load`.

    Attributes
    ----------
    c : float, optional (default=0.01)
        Regularization parameter. Higher values penalize longer rulelists.

    n_iter : int, optional (default=1000)
        Maximum number of nodes (rulelists) to search before exiting.
        Can be used to limit the memory footprint of faircorels.

    map_type : str, optional (default="prefix")
        The type of prefix map to use. Supported maps are "none" for no map,
        "prefix" for a map that uses rule prefixes for keys, "captured" for
        a map with a prefix's captured vector as keys.

    policy : str, optional (default="lower_bound")
        The search policy for traversing the tree (i.e. the criterion with which
        to order nodes in the queue). Supported criteria are "bfs", for breadth-first
        search; "curious", which attempts to find the most promising node; 
        "lower_bound" which is the objective function evaluated with that rulelist
        minus the default prediction error; "objective" for the objective function
        evaluated at that rulelist; and "dfs" for depth-first search.

    verbosity : list, optional (default=["rulelist"])
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

    ablation : int, optional (default=0)
        Specifies addition parameters for the bounds used while searching. Accepted
        values are 0 (all bounds), 1 (no antecedent support bound), and 2 (no
        lookahead bound).

    max_card : int, optional (default=2)
        Maximum cardinality allowed when mining rules. Can be any value greater than
        or equal to 1. For instance, a value of 2 would only allow rules that combine
        at most two features in their antecedents.

    min_support : float, optional (default=0.01)
        The fraction of samples that a rule must capture in order to be used. 1 minus
        this value is also the maximum fraction of samples a rule can capture.
        Can be any value between 0.0 and 0.5.
    
    beta : float, optional (default=0.0)
        The weight the unfairness in the objective function
    
    fairness: int optional (default=1)
        The type of fairness metric used. 
        1 : statistical parity, 2 : predictive parity, 3 : predictive equality, 4 : equal opportunity, 5 : equalized odds, 6 : conditional use accuracy equality

    maj_pos: int optional (default=-1)
        The position of the rule that defined the majority group
        If not specified, all individuals not in minority group are in majority group
        Not used if maj_vect is used

    min_pos: int optional (default=2)
        The position of the rule that defined the minority group
        Not used if min_vect is used

    maj_vect: int list optional (default=[-1])
        List of integers in {0,1} indicating if instances belong to the majority group
        If not specified, this vector is computed using maj_pos

    min_vect: int list optional (default=[-1])
        List of integers in {0,1} indicating if instances belong to the minority group
        If not specified, this vector is computed using min_pos

    mode: int optional (default=3)
        Method used for the multi-ojective framework
        1: weigted sum, 2: maximum fairness, 3: epsilon-constraint, 4: maximum accuracy

    useUnfairnessLB: bool optional (default=False)
        Use the unfairness lower bound
        -> Use the CP filtering when it is implemented, else uses a simple lower bound
    
    epsilon: float optional (default=0.05)
        max acceptable unfairness
    
    kbest: int optional (default=1)
        Randomly use one of the k best objectives
    
    forbidSensAttr: bool optional (default=False)
        Forbid the use of the sensitive and (if specified) unsensitive attributes in the produced rule lists

    bfs_mode: int optinal (default=0)
        BFS strategy
        0: original CORELS, 1:FIFO, 2:objective_aware, 3:lower_bound, 4:random

    random_state: int optional (default=42)
        Random seed for randomized search

    Arguments for .fit :

    performRestarts : int optional (default=0)
        0 : Does not perform any restarts
        1 : Performs geometric restarts
        2 : Performs Luby restarts

    initNBNodes : int optional (default=1000)
        Initial limit for the number of nodes in the trie when perfomring

    geomRReason : double optional (default=1.5)
        When performRestarts=True, geomRReason is the reason used 
        for the geometric restart calculation

    References
    ----------
    Elaine Angelino, Nicholas Larus-Stone, Daniel Alabi, Margo Seltzer, and Cynthia Rudin.
    Learning Certifiably Optimal Rule Lists for Categorical Data. KDD 2017.
    Journal of Machine Learning Research, 2018; 19: 1-77. arXiv:1704.01701, 2017

    Examples
    --------
    >>> import numpy as np
    >>> from corels import CorelsClassifier
    >>> X = np.array([ [1, 0, 1], [0, 1, 0], [1, 1, 1] ])
    >>> y = np.array([ 1, 0, 1])
    >>> c = CorelsClassifier(verbosity=[])
    >>> c.fit(X, y)
    ...
    >>> print(c.predict(X))
    [ True False  True ]
    """
    
    _estimator_type = "classifier"

    def __init__(self, c=0.01, n_iter=10000, map_type="prefix", policy="lower_bound",
                 verbosity=["rulelist"], ablation=0, max_card=2, min_support=0.01,
                 beta=0.0, fairness=1, maj_pos=-1, min_pos=2, maj_vect = np.empty(shape=(0)), min_vect = np.empty(shape=(0)),
                 mode=4, useUnfairnessLB=False, epsilon=0.0, kbest=1, forbidSensAttr=False,
                 bfs_mode=0, random_state=42):
        self.c = c
        self.n_iter = n_iter
        self.map_type = map_type
        self.policy = policy
        self.verbosity = verbosity
        self.ablation = ablation
        self.max_card = max_card
        self.min_support = min_support
        self.forbidSensAttr=forbidSensAttr
        self.beta = beta
        self.fairness = fairness
        if(maj_vect.size == 0):
            # Majority group is not explicitely defined
            # We will have to use maj_pos to compute the associated vector
            self.maj_pos = maj_pos
            if(maj_pos == -1):
                self.maj_vect = []
            #if(maj_pos != -1):
                #print("maj vect not specified, position ", maj_pos, " will be used.")
            #else:
                #print("no majority group defined, maj group will be all instances except minority group ones.")
                #self.maj_vect = []
        else:
            self.maj_pos = -2
            #print("maj vect specified")
            maj_vect = check_array(maj_vect, ndim=1)
            maj_vect = np.stack([ np.invert(maj_vect), maj_vect ])
            self.maj_vect = maj_vect


        if(min_vect.size == 0):
            # Majority group is not explicitely defined
            # We will have to use maj_pos to compute the associated vector
            self.min_pos = min_pos
            #print("min vect not specified, position ", min_pos, " will be used.")
        else:
            self.min_pos = -2
            min_vect = check_array(min_vect, ndim=1)
            min_vect = np.stack([ np.invert(min_vect), min_vect ])
            self.min_vect = min_vect
            #print("min vect specified")
        self.mode = mode
        self.useUnfairnessLB = useUnfairnessLB
        self.epsilon = epsilon
        self.kbest = kbest
        self.bfs_mode = bfs_mode
        self.random_state = random_state

    def fit(self, X, y, features=[], prediction_name="prediction", performRestarts=0, initNBNodes=1000, geomRReason=1.5, max_evals=1000000000, time_limit = None):
        """
        Build a CORELS classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8.

        y : array-line, shape = [n_samples]
            The target values for the training input. Must be binary.
        
        features : list, optional(default=[])
            A list of strings of length n_features. Specifies the names of each
            of the features. If an empty list is provided, the feature names
            are set to the default of ["feature1", "feature2"... ].

        prediction_name : string, optional(default="prediction")
            The name of the feature that is being predicted.

        max_evals : int, maximum number of calls to evaluate_children (ie maximum number of nodes explored in the prefix tree)

        time_limit : int, maximum number of seconds allowed for the model building
        Note that this specifies the CPU time and NOT THE WALL-CLOCK TIME

        Returns
        -------
        self : obj
        """
        if not isinstance(self.c, float):
            raise TypeError("Regularization constant (c) must be a float, got: " + str(type(self.c)))
        if self.c < 0.0 or self.c > 1.0:
            raise ValueError("Regularization constant (c) must be between 0.0 and 1.0, got: " + str(self.c))
        if not isinstance(self.n_iter, int):
            raise TypeError("Max nodes must be an integer, got: " + str(type(self.n_iter)))
        if self.n_iter < 0:
            raise ValueError("Max nodes must be positive, got: " + str(self.n_iter))
        if not isinstance(self.ablation, int):
            raise TypeError("Ablation must be an integer, got: " + str(type(self.ablation)))
        if self.ablation > 2 or self.ablation < 0:
            raise ValueError("Ablation must be between 0 and 2, inclusive, got: " + str(self.ablation))
        if not isinstance(self.map_type, str):
            raise TypeError("Map type must be a string, got: " + str(type(self.map_type)))
        if not isinstance(self.policy, str):
            raise TypeError("Policy must be a string, got: " + str(type(self.policy)))
        if not isinstance(self.verbosity, list):
            raise TypeError("Verbosity must be a list of strings, got: " + str(type(self.verbosity)))
        if not isinstance(self.min_support, float):
            raise TypeError("Minimum support must be a float, got: " + str(type(self.min_support)))
        if self.min_support < 0.0 or self.min_support > 0.5:
            raise ValueError("Minimum support must be between 0.0 and 0.5, got: " + str(self.min_support))
        if not isinstance(self.max_card, int):
            raise TypeError("Max cardinality must be an integer, got: " + str(type(self.max_card)))
        if self.max_card < 1:
            raise ValueError("Max cardinality must be greater than or equal to 1, got: " + str(self.max_card))
        if not isinstance(prediction_name, str):
            raise TypeError("Prediction name must be a string, got: " + str(type(prediction_name)))
        # Fairness params
        if not isinstance(self.beta, float):
            raise TypeError("Unfairness weight (beta) must be a float, got: " + str(type(self.beta)))
        if not isinstance(self.fairness, int):
            raise TypeError("Fairness metric id must be an integer between 1 and 4, got: " + str(type(self.fairness)))
        if self.fairness < 1 or self.fairness > 6:
            raise ValueError("Fairness metric id must be an integer between 1 and 4, got: " + str((self.fairness)))
        if not isinstance(self.maj_pos, int):
            raise TypeError("The position maj_pos of the rule that defined the majority group  must be an integer, got: " + str(type(self.maj_pos)))
        if not isinstance(self.min_pos, int):
            raise TypeError("The position min_pos of the rule that defined the minority group  must be an integer, got: " + str(type(self.min_pos)))

        # Todo check mode, .... (?)
       
        if(self.min_pos != -2):
            min_vect = X[:,self.min_pos]
            min_vect = check_array(min_vect, ndim=1)
            min_vect = np.stack([ np.invert(min_vect), min_vect ])
            self.min_vect = min_vect
        #print(len(self.min_vect), " elements in min_vect, %d captured" %(self.min_vect.count(1)))
        if(self.maj_pos != -2):
            if self.maj_pos == -1: # Nor vector for majority group given neither column number => all instances not in min group are in maj group
                self.maj_vect = np.empty(shape=(self.min_vect.shape))
                for e in range(self.min_vect.shape[1]):
                    if self.min_vect[0][e] == 1:
                        self.maj_vect[0][e] = 0
                        self.maj_vect[1][e] = 1
                    else:
                        self.maj_vect[0][e] = 1
                        self.maj_vect[1][e] = 0
            else:
                maj_vect =  X[:,self.maj_pos]
                maj_vect = check_array(maj_vect, ndim=1)
                maj_vect = np.stack([ np.invert(maj_vect), maj_vect ])
                self.maj_vect = maj_vect

        #print(len(self.maj_vect), " elements in maj_vect, %d captured" %(self.maj_vect.count(1)))
        label = check_array(y, ndim=1)
        labels = np.stack([ np.invert(label), label ])
        samples = check_array(X, ndim=2)
        check_consistent_length(samples, labels)

        n_samples = samples.shape[0]
        n_features = samples.shape[1]
        if self.max_card > n_features:
            raise ValueError("Max cardinality (" + str(self.max_card) + ") cannot be greater"
                             " than the number of features (" + str(n_features) + ")")

        n_labels = labels.shape[0]
        
        rl = RuleList()
        
        if features:
            check_features(features)
            rl.features = list(features)
        else:
            rl.features = []
            for i in range(n_features):
                rl.features.append("feature" + str(i + 1))

        if rl.features and len(rl.features) != n_features:
            raise ValueError("Feature count mismatch between sample data (" + str(n_features) + 
                             ") and feature names (" + str(len(rl.features)) + ")")
        
        rl.prediction_name = prediction_name

        allowed_verbosities = ["rulelist", "rule", "label", "samples", "progress", "loud", "mine", "minor"]
        for v in self.verbosity:
            if not isinstance(v, str):
                raise TypeError("Verbosity flags must be strings, got: " + str(v))

            check_in("Verbosities", allowed_verbosities, v)
        
        if "samples" in self.verbosity \
              and "rule" not in self.verbosity \
              and "label" not in self.verbosity \
              and "minor" not in self.verbosity \
              and "loud" not in self.verbosity:
            raise ValueError("'samples' verbosity option must be combined with at" + 
                             " least one of 'rule', 'label', 'minor', or 'loud'")

        # Verbosity for rule mining and minority bound. 0 is quiet, 1 is verbose
        mine_verbose = 0
        if "loud" in self.verbosity or "mine" in self.verbosity:
            mine_verbose = 1
        
        minor_verbose = 0
        if "loud" in self.verbosity or "minor" in self.verbosity:
            minor_verbose = 1
        
        verbose = ",".join([ v for v in self.verbosity if v != "rulelist" ])

        map_types = ["none", "prefix", "captured"]
        policies = ["bfs", "curious", "lower_bound", "objective", "dfs"]

        check_in("Map type", map_types, self.map_type)
        check_in("Search policy", policies, self.policy)

        map_id = map_types.index(self.map_type)
        policy_id = policies.index(self.policy)
        accuracy_upper_bound = computeAccuracyUpperBound(X, y, verbose=0)
        fr = fit_wrap_begin(samples.astype(np.uint8, copy=False),
                             labels.astype(np.uint8, copy=False), rl.features,
                             self.max_card, self.min_support, verbose, mine_verbose, minor_verbose,
                             self.c, policy_id, map_id, self.ablation, False, self.forbidSensAttr, self.bfs_mode, self.random_state,
                             self.maj_vect.astype(np.uint8, copy=False), self.min_vect.astype(np.uint8, copy=False), accuracy_upper_bound, max_evals)
        
        if fr:
            early = False
            try:
                if time_limit is None: 
                    while fit_wrap_loop(self.n_iter, self.beta, self.fairness, self.mode, self.useUnfairnessLB, self.epsilon, self.kbest, performRestarts, initNBNodes, geomRReason):
                        pass
                else:
                    import time
                    start = time.clock()
                    while fit_wrap_loop(self.n_iter, self.beta, self.fairness, self.mode, self.useUnfairnessLB, self.epsilon, self.kbest, performRestarts, initNBNodes, geomRReason):
                        end = time.clock()
                        if end - start > time_limit:
                            if debug:
                                print("Exiting because CPU time limit is reached (", end - start, " seconds / ", time_limit, ".")
                            break
            except:
                print("\nExiting early")
                rl.rules, self.nbExplored, self.nbCache = fit_wrap_end(True)
                self.rl_ = rl

                raise
             
            rl.rules, self.nbExplored, self.nbCache = fit_wrap_end(False)
            self.rl_ = rl

            if "rulelist" in self.verbosity:
                print(self.rl_)
        else:
            print("Error running model! Exiting")
        return self

    def predict(self, X):
        """
        Predict classifications of the input samples X.

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8. The features must be the same
            as those of the data used to train the model.

        Returns
        -------
        p : array of shape = [n_samples].
            The classifications of the input samples.
        """
        check_is_fitted(self, "rl_")
        check_rulelist(self.rl_)        

        samples = check_array(X, ndim=2)
        
        if samples.shape[1] != len(self.rl_.features):
            raise ValueError("Feature count mismatch between eval data (" + str(X.shape[1]) + 
                             ") and feature names (" + str(len(self.rl_.features)) + ")")
        return np.array(predict_wrap(samples.astype(np.uint8, copy=False), self.rl_.rules), dtype=np.int32)
                
    def predict_with_scores(self, X):
        """
        Predict classifications of the input samples X.

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8. The features must be the same
            as those of the data used to train the model.

        Returns
        -------
        p : array of shape = [[n_samples],[n_samples]].
            The first array contains the classifications of the input samples.
            The second array contains the associated confidence scores.
        """

        check_is_fitted(self, "rl_")
        check_rulelist(self.rl_)        

        samples = check_array(X, ndim=2)
        
        if samples.shape[1] != len(self.rl_.features):
            raise ValueError("Feature count mismatch between eval data (" + str(X.shape[1]) + 
                             ") and feature names (" + str(len(self.rl_.features)) + ")")
        preds, scores = predict_score_wrap(samples.astype(np.uint8, copy=False), self.rl_.rules)
        predsArray = np.array(preds, dtype=np.int32)
        scoresArray = np.array(scores, dtype=np.double)
        return np.array((predsArray, scoresArray))
    
    def score(self, X, y):
        """
        Score the algorithm on the input samples X with the labels y. Alternatively,
        score the predictions X against the labels y (where X has been generated by 
        `predict` or something similar).

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features] OR shape = [n_samples]
            The input samples, or the sample predictions. All features must be binary.
        
        y : array-like, shape = [n_samples]
            The input labels. All labels must be binary.

        Returns
        -------
        a : float
            The accuracy, from 0.0 to 1.0, of the rulelist predictions
        """

        labels = check_array(y, ndim=1)
        p = check_array(X)
        check_consistent_length(p, labels)
        
        if p.ndim == 2:
            p = self.predict(p)
        elif p.ndim != 1:
            raise ValueError("Input samples must have only 1 or 2 dimensions, got " + str(p.ndim) +
                             " dimensions")

        a = np.mean(np.invert(np.logical_xor(p, labels)))

        return a

    def get_params(self):
        """
        Get a list of all the model's parameters.
        
        Returns
        -------
        params : dict
            Dictionary of all parameters, with the names of the parameters as the keys
        """

        return {
            "c": self.c,
            "n_iter": self.n_iter,
            "map_type": self.map_type,
            "policy": self.policy,
            "verbosity": self.verbosity,
            "ablation": self.ablation,
            "max_card": self.max_card,
            "min_support": self.min_support,
            "beta": self.beta,
            "fairness": self.fairness,
            "maj_pos": self.maj_pos,
            "min_pos": self.min_pos,
            "maj_vect": self.maj_vect,
            "min_vect": self.min_vect,
            "mode": self.mode,
            "useUnfairnessLB": self.useUnfairnessLB,
            "epsilon": self.epsilon,
            "kbest": self.kbest,
            "bfs_mode": self.bfs_mode,
            "random_sate": self.random_state
        }

    def set_params(self, **params):
        """
        Set model parameters. Takes an arbitrary number of keyword parameters, all of which
        must be valid parameter names (i.e. must be included in those returned by get_params).

        Returns
        -------
        self : obj
        """
        
        valid_params = self.get_params().keys()

        for param, val in params.items():
            if param not in valid_params:
                raise ValueError("Invalid parameter '" + str(param) + "' given in set_params. "
                                 "Check the list of valid parameters with get_params()")
            setattr(self, param, val)

        return self

    def save(self, fname):
        """
        Save the model to a file, using python's pickle module.

        Parameters
        ----------
        fname : string
            File name to store the model in
        
        Returns
        -------
        self : obj
        """

        with open(fname, "wb") as f:
            pickle.dump(self, f)

        return self

    def load(self, fname):
        """
        Load a model from a file, using python's pickle module.
        
        Parameters
        ----------
        fname : string
            File name to load the model from
        
        Returns
        -------
        self : obj
        """

        with open(fname, "rb") as f:
            model = pickle.load(f)
           
            if not hasattr(model, "get_params"):
                raise ValueError("Invalid model provided, model must have get_params() method")
                
            self.set_params(**model.get_params())

            if hasattr(model, "rl_"):
                self.rl_ = model.rl_

        return self

    def rl(self, set_val=None):
        """
        Return or set the learned rulelist
        
        Parameters
        ----------
        set_val : RuleList, optional
            Rulelist to set the model to

        Returns
        -------
        rl : obj
            The model's rulelist
        """

        if set_val != None:
            check_rulelist(set_val)

            self.rl_ = set_val
        else:
            check_is_fitted(self, "rl_")
        
        return self.rl_
    
    def __str__(self):
        s = "CorelsClassifier (" + str(self.get_params()) + ")"

        if hasattr(self, "rl_"):
            s += "\n" + self.rl_.__str__()

        return s
    
    def __repr__(self):
        s = "CorelsClassifier (" + str(self.get_params()) + ")"

        if hasattr(self, "rl_"):
            s += "\n" + self.rl_.__repr__()

        return s

    def explain(self, anEx):
        """
        Explains a prediction
        Arguments
        ---------
        anEx : array-like, shape = [n_features] 
            The input sample
        
        Returns
        -------
        a : list l where
            l[0] is the instance's prediction
            l[1] is the implicant(s) that led to that decision
            (both are strings - user friendly)
        """
        if len(self.rl_.rules) == 1:
            return [self.predict([anEx]), "DEFAULT DECISION -> ", self.rl_.prediction_name + " = " + str(self.rl_.rules[0]["prediction"])] #+ " (conf score = " + str(self.rules[0]["score"]) + ")"
        else:    
            for i in range(len(self.rl_.rules) - 1):
                match = True
                feat = get_feature(self.rl_.features, self.rl_.rules[i]["antecedents"][0])
                #print(anEx[self.rl_.rules[i]["antecedents"][0]-1] )
                if anEx[self.rl_.rules[i]["antecedents"][0]-1] != 1:
                    match = False
                for j in range(1, len(self.rl_.rules[i]["antecedents"])):
                    feat += " && " + get_feature(self.rl_.features, self.rl_.rules[i]["antecedents"][j])
                    if anEx[ self.rl_.rules[i]["antecedents"][j]-1] != 1:
                        match = False
                if match:
                    return [self.predict([anEx]), "[" + feat + "]-> " + self.rl_.prediction_name + " = " + str(bool(self.rl_.rules[i]["prediction"]))] # + " (conf score = " + str(self.rules[i]["score"]) + ")"
            return [self.predict([anEx]), "DEFAULT DECISION -> ", self.rl_.prediction_name + " = " + str(self.rl_.rules[-1]["prediction"])]# + " (conf score = " + str(str(self.rules[-1]["score"])) + ")"

    def explain_api(self, anEx):
        """
        Explains a prediction (shorter output)
        Arguments
        ---------
        anEx : array-like, shape = [n_features] 
            The input sample
        
        Returns
        -------
        a : list l where
            l[0] is the instance's prediction
            l[1] is the implicant(s) that led to that decision
            (both are API-oriented - easy to use by a program)
        """
        if len(self.rl_.rules) == 1:
            return [self.rl_.rules[0]["prediction"], "DEFAULT DECISION"]
        else:    
            for i in range(len(self.rl_.rules) - 1):
                match = True
                feat = get_feature(self.rl_.features, self.rl_.rules[i]["antecedents"][0])
                if anEx[self.rl_.rules[i]["antecedents"][0]-1] != 1:
                    match = False
                for j in range(1, len(self.rl_.rules[i]["antecedents"])):
                    feat += " && " + get_feature(self.rl_.features, self.rl_.rules[i]["antecedents"][j])
                    if anEx[ self.rl_.rules[i]["antecedents"][j]-1] != 1:
                        match = False
                if match:
                    return [self.rl_.rules[i]["prediction"],feat]
            return [self.rl_.rules[-1]["prediction"], "DEFAULT DECISION"]

    def explain_long(self, anEx):
        """
        Explains a prediction
        Adds by the negation of previous implicants, when applicable
        Arguments
        ---------
        anEx : array-like, shape = [n_features] 
            The input sample
        
        Returns
        -------
        a : list l where
            l[0] is the instance's prediction
            l[1] is the implicant(s) that led to that decision
            (both are strings - user friendly)
        """
        if len(self.rl_.rules) == 1:
            return [self.predict([anEx]), "DEFAULT DECISION -> ", self.rl_.prediction_name + " = " + str(self.rl_.rules[0]["prediction"])] #+ " (conf score = " + str(self.rules[0]["score"]) + ")"
        else:    
            neg = ""
            for i in range(len(self.rl_.rules) - 1):
                match = True
                feat = get_feature(self.rl_.features, self.rl_.rules[i]["antecedents"][0])
                #print(anEx[self.rl_.rules[i]["antecedents"][0]-1] )
                if anEx[self.rl_.rules[i]["antecedents"][0]-1] != 1:
                    match = False
                for j in range(1, len(self.rl_.rules[i]["antecedents"])):
                    feat += " && " + get_feature(self.rl_.features, self.rl_.rules[i]["antecedents"][j])
                    if anEx[ self.rl_.rules[i]["antecedents"][j]-1] != 1:
                        match = False
                if match:
                    if len(neg) > 0:
                        neg += " AND "
                    return [self.predict([anEx]), "[" + neg + feat + "]-> " + self.rl_.prediction_name + " = " + str(bool(self.rl_.rules[i]["prediction"]))] # + " (conf score = " + str(self.rules[i]["score"]) + ")"
                else:
                    if len(neg) == 0:
                        neg+="NOT [%s]" %feat
                    else:
                        neg+=" AND NOT [%s]" %feat
            return [self.predict([anEx]), neg, self.rl_.prediction_name + " = " + str(self.rl_.rules[-1]["prediction"])]# + " (conf score = " + str(str(self.rules[-1]["score"])) + ")"

    def explain_long_api(self, anEx):
            """
            Explains a prediction (shorter output)
            Adds by the negation of previous implicants, when applicable
            Arguments
            ---------
            anEx : array-like, shape = [n_features] 
                The input sample
            
            Returns
            -------
            a : list l where
                l[0] is the instance's prediction
                l[1] is the implicant(s) (and previous negations) that led to that decision
                (both are API-oriented - easy to use by a program)
            """
            if len(self.rl_.rules) == 1:
                return [self.rl_.rules[0]["prediction"], "DEFAULT DECISION"]
            else:    
                neg = ""
                for i in range(len(self.rl_.rules) - 1):
                    match = True
                    feat = get_feature(self.rl_.features, self.rl_.rules[i]["antecedents"][0])
                    if anEx[self.rl_.rules[i]["antecedents"][0]-1] != 1:
                        match = False
                    for j in range(1, len(self.rl_.rules[i]["antecedents"])):
                        feat += " && " + get_feature(self.rl_.features, self.rl_.rules[i]["antecedents"][j])
                        if anEx[ self.rl_.rules[i]["antecedents"][j]-1] != 1:
                            match = False
                    if match:
                        if len(neg) > 0:
                            neg += " AND "
                        return [self.rl_.rules[i]["prediction"],neg+feat]
                    else:
                        if len(neg) == 0:
                            neg+="NOT [%s]" %feat
                        else:
                            neg+=" AND NOT [%s]" %feat
                return [self.rl_.rules[-1]["prediction"], neg]

class CorelsBagging:
    """
    Attributes
    ----------
    n_learners : # base learners
    sample_size : size of one subsample (used to train exactly one base learner)
    + all CorelsClassifier attributes used to create the base learners
    """
    _estimator_type = "classifier"

    def __init__(self, X, y, n_learners, sample_size, features=[], prediction_name="prediction", c=0.01, n_iter=10000, map_type="prefix", policy="lower_bound",
                 verbosity=["rulelist"], ablation=0, max_card=2, min_support=0.01,
                 beta=0.0, fairness=1, maj_pos=-1, min_pos=2, maj_vect = np.empty(shape=(0)), min_vect = np.empty(shape=(0)),
                 mode=4, useUnfairnessLB=False, epsilon=0.0, kbest=1, forbidSensAttr=False,
                 bfs_mode=0, random_state=42, baggingVerbose=0):   
        # Check arguments
        if not isinstance(n_learners, int):
            raise TypeError("n_learners must be an integer, got: " + str(type(self.n_learners)))
        if n_learners < 0:
            raise ValueError("n_learners must be strictly greater than 0, got: " + str(self.n_learners)) 
        if not isinstance(sample_size, int):
            raise TypeError("sample_size must be an integer, got: " + str(type(self.sample_size)))
        if sample_size < 0:
            raise ValueError("sample_size must be strictly greater than 0, got: " + str(self.sample_size)) 
        
        # Ensemble parameters
        self.n_learners = n_learners
        self.sample_size = sample_size

        # Other parameters assignment and check
        self.X = X
        self.y = y
        self.features = features
        self.c = c
        self.n_iter = n_iter
        self.map_type = map_type
        self.policy = policy
        self.verbosity = verbosity
        self.ablation = ablation
        self.max_card = max_card
        self.min_support = min_support
        self.forbidSensAttr=forbidSensAttr
        self.beta = beta
        self.fairness = fairness
        self.prediction_name = prediction_name
        if not isinstance(self.c, float):
            raise TypeError("Regularization constant (c) must be a float, got: " + str(type(self.c)))
        if self.c < 0.0 or self.c > 1.0:
            raise ValueError("Regularization constant (c) must be between 0.0 and 1.0, got: " + str(self.c))
        if not isinstance(self.n_iter, int):
            raise TypeError("Max nodes must be an integer, got: " + str(type(self.n_iter)))
        if self.n_iter < 0:
            raise ValueError("Max nodes must be positive, got: " + str(self.n_iter))
        if not isinstance(self.ablation, int):
            raise TypeError("Ablation must be an integer, got: " + str(type(self.ablation)))
        if self.ablation > 2 or self.ablation < 0:
            raise ValueError("Ablation must be between 0 and 2, inclusive, got: " + str(self.ablation))
        if not isinstance(self.map_type, str):
            raise TypeError("Map type must be a string, got: " + str(type(self.map_type)))
        if not isinstance(self.policy, str):
            raise TypeError("Policy must be a string, got: " + str(type(self.policy)))
        if not isinstance(self.verbosity, list):
            raise TypeError("Verbosity must be a list of strings, got: " + str(type(self.verbosity)))
        if not isinstance(self.min_support, float):
            raise TypeError("Minimum support must be a float, got: " + str(type(self.min_support)))
        if self.min_support < 0.0 or self.min_support > 0.5:
            raise ValueError("Minimum support must be between 0.0 and 0.5, got: " + str(self.min_support))
        if not isinstance(self.max_card, int):
            raise TypeError("Max cardinality must be an integer, got: " + str(type(self.max_card)))
        if self.max_card < 1:
            raise ValueError("Max cardinality must be greater than or equal to 1, got: " + str(self.max_card))
        if not isinstance(prediction_name, str):
            raise TypeError("Prediction name must be a string, got: " + str(type(prediction_name)))
        # Fairness params
        if not isinstance(self.beta, float):
            raise TypeError("Unfairness weight (beta) must be a float, got: " + str(type(self.beta)))
        if not isinstance(self.fairness, int):
            raise TypeError("Fairness metric id must be an integer between 1 and 4, got: " + str(type(self.fairness)))
        if self.fairness < 1 or self.fairness > 6:
            raise ValueError("Fairness metric id must be an integer between 1 and 4, got: " + str((self.fairness)))
        if not isinstance(maj_pos, int):
            raise TypeError("The position maj_pos of the rule that defined the majority group  must be an integer, got: " + str(type(maj_pos)))
        if not isinstance(min_pos, int):
            raise TypeError("The position min_pos of the rule that defined the minority group  must be an integer, got: " + str(type(min_pos)))

        self.mode = mode
        self.useUnfairnessLB = useUnfairnessLB
        self.epsilon = epsilon
        self.kbest = kbest
        self.bfs_mode = bfs_mode
        self.random_state = random_state
        self.baggingVerbose = baggingVerbose
        # Prepare min_vect and maj_vect for entire original training set whatever args were given
        if(maj_vect.size == 0):
            # Majority group is not explicitely defined
            # We will have to use maj_pos to compute the associated vector
            self.maj_pos = maj_pos
            if(maj_pos == -1):
                self.maj_vect = []
            #if(maj_pos != -1):
                #print("maj vect not specified, position ", maj_pos, " will be used.")
            #else:
                #print("no majority group defined, maj group will be all instances except minority group ones.")
                #self.maj_vect = []
        else:
            self.maj_pos = -2
            #print("maj vect specified")
            maj_vect = check_array(maj_vect, ndim=1)
            maj_vect = np.stack([ np.invert(maj_vect), maj_vect ])
            self.maj_vect = maj_vect


        if(min_vect.size == 0):
            # Majority group is not explicitely defined
            # We will have to use maj_pos to compute the associated vector
            self.min_pos = min_pos
            #print("min vect not specified, position ", min_pos, " will be used.")
        else:
            self.min_pos = -2
            min_vect = check_array(min_vect, ndim=1)
            min_vect = np.stack([ np.invert(min_vect), min_vect ])
            self.min_vect = min_vect
            #print("min vect specified")

        if(self.min_pos != -2):
            min_vect = X[:,self.min_pos]
            min_vect = check_array(min_vect, ndim=1)
            min_vect = np.stack([ np.invert(min_vect), min_vect ])
            self.min_vect = min_vect
        #print(len(self.min_vect), " elements in min_vect, %d captured" %(self.min_vect.count(1)))
        if(self.maj_pos != -2):
            if self.maj_pos == -1: # Nor vector for majority group given neither column number => all instances not in min group are in maj group
                self.maj_vect = np.empty(shape=(self.min_vect.shape))
                for e in range(self.min_vect.shape[1]):
                    if self.min_vect[0][e] == 1:
                        self.maj_vect[0][e] = 0
                        self.maj_vect[1][e] = 1
                    else:
                        self.maj_vect[0][e] = 1
                        self.maj_vect[1][e] = 0
            else:
                maj_vect =  X[:,self.maj_pos]
                maj_vect = check_array(maj_vect, ndim=1)
                maj_vect = np.stack([ np.invert(maj_vect), maj_vect ])
                self.maj_vect = maj_vect

  
        self.X_sets = []
        self.Y_sets = []
        self.maj_sets = []
        self.min_sets = []
        self.learners = []
        captTab = []
        if self.baggingVerbose > 0:
            print("[BAGGING - MAIN] Initializing ", self.n_learners, " training sets of size ", self.sample_size, ".")
        np.random.seed(self.random_state+1) # so that the random_state used inside corels is different
        for i in range(self.n_learners):
            indices = np.random.randint(low=0, high=self.X.shape[0], size=self.sample_size)  # np.asarray(range(self.X.shape[0])) # to keep X identical for all base learners
            self.X_sets.append(self.X[indices])
            self.Y_sets.append(self.y[indices])
            captTab.append(np.shape(np.unique(indices))[0])
            #print(self.maj_vect.shape)
            #print(indices)
            temp_maj_vect = self.maj_vect[0][indices]
            #self.maj_sets.append(np.stack([ np.invert(temp_maj_vect), temp_maj_vect ]))
            self.maj_sets.append(temp_maj_vect)
            temp_min_vect = self.min_vect[0][indices]
            #self.min_sets.append(np.stack([ np.invert(temp_min_vect), temp_min_vect ]))
            self.min_sets.append(temp_min_vect)
        if self.baggingVerbose > 0:
            captAv = np.average(captTab)
            print("[BAGGING - MAIN] prepared bags contain average of %f/%d (%f) of original instances each." %(captAv, self.X.shape[0], captAv/self.X.shape[0]))
            print("[BAGGING - MAIN] Initializing ", self.n_learners, " learners.")
        for i in range(self.n_learners):
            self.learners.append(CorelsClassifier(c=self.c, n_iter=self.n_iter, map_type=self.map_type, policy=self.policy,
                 verbosity=self.verbosity, ablation=self.ablation, max_card=self.max_card, min_support=self.min_support,
                 beta=self.beta, fairness=self.fairness, 
                 maj_pos=self.maj_pos, 
                 min_pos=self.min_pos, 
                 maj_vect = self.maj_sets[i], 
                 min_vect = self.min_sets[i],
                 mode=self.mode, useUnfairnessLB=self.useUnfairnessLB, epsilon=self.epsilon, kbest=self.kbest, forbidSensAttr=self.forbidSensAttr,
                 bfs_mode=self.bfs_mode, random_state=self.random_state))
        if self.baggingVerbose > 0:
            print("Initialized learners")

    def train_one(self, modelIndex):
        my_learner = self.learners[modelIndex]
        my_learner.fit(self.X_sets[modelIndex], self.Y_sets[modelIndex], features=self.features, prediction_name=self.prediction_name, 
        performRestarts=self.performRestarts, initNBNodes=self.initNBNodes, geomRReason = self.geomRReason, max_evals=self.max_evals, time_limit = self.time_limit)
        return my_learner

    def fit(self, performRestarts=0, initNBNodes=1000, geomRReason=1.5, max_evals=1000000000, time_limit = None, n_workers=-1):
        """
        Trains all the base learners on their respective bags

        Parameters
        ----------
        n_workers : int, maximum number of threads allowed to parallelize the training of the different learners

        max_evals : int, maximum number of calls to evaluate_children (ie maximum number of nodes explored in the prefix tree)

        time_limit : int, maximum number of seconds allowed for the model building
        Note that this specifies the CPU time and NOT THE WALL-CLOCK TIME

        + restart parameters (see above in CorelsClassifier doc)

        Returns
        -------
        self : obj
        """
        self.performRestarts=performRestarts
        self.initNBNodes=initNBNodes
        self.geomRReason=geomRReason
        self.max_evals=max_evals
        self.time_limit = time_limit
        if self.baggingVerbose > 0:
            print("[BAGGING - MAIN] Training ", self.n_learners, " learners (using at most %d threads)." %n_workers)
        trained_learners = Parallel(n_jobs=n_workers)(delayed(self.train_one)(modelIndex) for modelIndex in tqdm(range(self.n_learners)))
        self.learners = trained_learners
        if self.baggingVerbose > 0:
            print("[BAGGING - MAIN] Trained ", self.n_learners, " learners.")
        ruleLists = []
        nbExploredList = []
        nbCacheList = []
        for i in range(self.n_learners):
            ruleLists.append(self.learners[i].rl_)
            nbExploredList.append(self.learners[i].nbExplored)
            nbCacheList.append(self.learners[i].nbCache)
        
        self.ruleLists = ruleLists
        self.nbExploredList = nbExploredList
        self.nbCacheList = nbCacheList
        # --- the remaining of this function is only a redundant check for debugging ---
        # check that each model meets unf constraint on its training set
        #print("epsilon = ", self.epsilon)
        import pandas as pd
        for i in range(self.n_learners):
            df_train = pd.DataFrame(self.X_sets[i], columns=self.features)
            df_train['two_year_recid'] = self.Y_sets[i]
            df_train["predictions"] = self.learners[i].predict(self.X_sets[i])
            cm_train = ConfusionMatrix(self.min_sets[i], self.maj_sets[i], df_train["predictions"], df_train["two_year_recid"])
            cm_minority_train, cm_majority_train = cm_train.get_matrix()
            fm_train = Metric(cm_minority_train, cm_majority_train)

            if self.fairness == 1:
                unfTraining = fm_train.statistical_parity()
            elif self.fairness == 2:
                unfTraining = fm_train.predictive_parity()
            elif self.fairness == 3:
                unfTraining = fm_train.predictive_equality()
            elif self.fairness == 4:
                unfTraining = fm_train.equal_opportunity()
            else:
                unfTraining = -1
            #print("unf = ", unfTraining)
            if unfTraining > 1 - self.epsilon:
                print("ERROR unf too high on his own training set ! :o")
                raise ValueError

    def majority(self, n):
        """
        Voting function (majority wins)
        Also updates the tie global counter
        """
        #print("[BAGGING - MAIN] Aggregator got %d votes 1 and %d votes 0." %(n, self.n_learners-n))
        if n > (self.n_learners-n):
            return 1
        else:
            if n == (self.n_learners-n):
                self.num_ties += 1
            return 0

    def predict(self, X):
        """
        Predict classifications of the input samples X.

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. All features must be binary, and the matrix
            is internally converted to dtype=np.uint8. The features must be the same
            as those of the data used to train the model.

        Returns
        -------
        p : array of shape = [n_samples].
            The classifications of the input samples.
        """
        first = True
        self.num_ties = 0
        for i in range(self.n_learners):
            if first:
                preds = self.learners[i].predict(X)
                first = False
            else:
                preds = np.add(preds, self.learners[i].predict(X))
        final_preds = np.asarray([self.majority(preds[i]) for i in range(preds.shape[0])])
        if self.baggingVerbose > 0:
            print("Aggregator had to break %d ties. (over %d total votes)" %(self.num_ties,final_preds.size))
        return final_preds
    
    def score(self, X, y):
        """
        Score the algorithm on the input samples X with the labels y. Alternatively,
        score the predictions X against the labels y (where X has been generated by 
        `predict` or something similar).

        Arguments
        ---------
        X : array-like, shape = [n_samples, n_features] OR shape = [n_samples]
            The input samples, or the sample predictions. All features must be binary.
        
        y : array-like, shape = [n_samples]
            The input labels. All labels must be binary.

        Returns
        -------
        a : float
            The accuracy, from 0.0 to 1.0, of the rulelist predictions
        """

        labels = check_array(y, ndim=1)
        p = check_array(X)
        check_consistent_length(p, labels)
        
        if p.ndim == 2:
            p = self.predict(p)
        elif p.ndim != 1:
            raise ValueError("Input samples must have only 1 or 2 dimensions, got " + str(p.ndim) +
                             " dimensions")

        a = np.mean(np.invert(np.logical_xor(p, labels)))

        return a

    def explain(self, anInst):
        """
        Explains a prediction
        Arguments
        ---------
        anInst : array-like, shape = [n_features]
            The input sample
        
        Returns
        -------
        d : dictionnary {'implicants':i, 'prediction':p}
            where i is the list of implicants that led majority voters to their prediction
            and p is the associated prediction
        """
        pred = self.predict([anInst])
        expls = []
        for i in range(self.n_learners):
            if self.learners[i].predict([anInst]) == pred:
                expl = self.learners[i].explain_api(anInst)
                if int(expl[0]) != pred[0]:
                    print("ERROR, redundant check failed :(")
                    raise ValueError
                expls.append(expl[1])
        return {'prediction': self.prediction_name + " = " + str(bool(pred[0])), 'implicants':list(dict.fromkeys(expls))}

    def explain_complete(self, anInst):
        """
        Explains a prediction (adds complete implications, including antecedents negation for all learners)
        Arguments
        ---------
        anInst : array-like, shape = [n_features]
            The input sample
        
        Returns
        -------
        d : dictionnary {'implicants':i, 'prediction':p}
            where i is the list of implicants that led majority voters to their prediction
            and p is the associated prediction
        """
        pred = self.predict([anInst])
        expls = []
        for i in range(self.n_learners):
            if self.learners[i].predict([anInst]) == pred:
                expl = self.learners[i].explain_long_api(anInst)
                if int(expl[0]) != pred[0]:
                    print("ERROR, redundant check failed :(")
                    raise ValueError
                expls.append(expl[1])
        return {'prediction': self.prediction_name + " = " + str(bool(pred[0])), 'implicants':list(dict.fromkeys(expls))}