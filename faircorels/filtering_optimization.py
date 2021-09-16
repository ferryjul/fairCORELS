from .filtering_optimization_cplex import run_filtering_opt_cplex

class OptimizationPruningWrapper():
    """
    Tool to perform MILP pruning. Uses memoisation.
    CPLEX (Python solver) used as backend.
    It has to be instanciated because it is able to use memoisation, and thus possibly avoid useless and costly calls to the solver.
    """

    def __init__(self, memoisation=1, fairnessMetric=1, fairness_tolerence=0.0, check_response=False, debug=0):
        self.memo = {}
        self.memoisation = memoisation
        if self.memoisation > 0:
            self.memoRead=0
            self.memoCheck=0
        self.fairnessMetric=fairnessMetric
        self.fairness_tolerence=fairness_tolerence
        self.check_response = check_response
        self.debug = debug
        print("Initialized pruning auditor with parameters: memoisation= %d, fairnessMetric= %d, fairness_tolerence= %f, check_response=%s" %(self.memoisation, self.fairnessMetric, self.fairness_tolerence, str(self.check_response)))

    """
    Returns:
    -1 if problem is UNSAT
    OR
    Maximum number of well classified examples (>0) for any extension, if problem is SAT.
    """
    def perform_opt_pruning(self, nb_sp_plus,	nb_sp_minus, nb_su_plus, nb_su_minus, L, U, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu):
        if self.memoisation == 1:
            key = tuple([nb_sp_plus, nb_sp_minus, nb_su_plus, nb_su_minus, L, U, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu])
            self.memoCheck+=1
            if key in self.memo:
                if self.debug >= 1:
                    print("Retrieving result saved in memo")
                self.memoRead+=1
                return self.memo[key]
        res = run_filtering_opt_cplex(self.fairnessMetric, nb_sp_plus, nb_sp_minus, nb_su_plus, nb_su_minus, L, U, self.fairness_tolerence, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu, self.check_response)
        if self.memoisation == 1:
            key = tuple([nb_sp_plus, nb_sp_minus, nb_su_plus, nb_su_minus, L, U, TPp, FPp, TNp, FNp, TPu, FPu, TNu, FNu])
            self.memo[key] = res
        return res

    """
    prints useful statistics about the use of memoisation
    Can be used to estimate the memory use and eventual gain induced by the use of the memoisation technique
    """
    def print_memo_statistics(self):
        if self.memoisation == 1:
            print("Simple memoisation is active.")
            print("     Memo contains %d elements, was checked %d times, avoided %d solver calls."%(len(self.memo), self.memoCheck, self.memoRead))
        else:
            print("Memoisation is not active.")

