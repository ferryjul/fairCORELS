def run_filtering_opt_cplex(fairnessMetric,
            nb_sp_plus,
			nb_sp_minus,
			nb_su_plus,
			nb_su_minus,
			L,
			U,
			fairness_tolerence,
			TPp,
			FPp,
			TNp,
			FNp,
			TPu,
			FPu,
			TNu,
			FNu,
            check_response):
    import cplex

    problem = cplex.Cplex()
    problem.set_log_stream(None)
    #prob.set_error_stream(None)
    #prob.set_warning_stream(None)
    problem.set_results_stream(None)
    #problem.parameters.threads = 1

    # We want to find a maximum of our objective function
    problem.objective.set_sense(problem.objective.sense.maximize)

    # The names of our variables
    names = ["sp_plus", "sp_minus", "su_plus", "su_minus"]

    # The obective function. More precisely, the coefficients of the objective
    # function. Note that we are casting to floats.
    objective = [1.0, -1.0, 1.0, -1.0]

    # Lower bounds. Since these are all zero, we could simply not pass them in as
    # all zeroes is the default.
    lower_bounds = [TPp, FPp, TPu, FPu]

    # Upper bounds. The default here would be cplex.infinity, or 1e+20.
    upper_bounds = [nb_sp_plus - FNp, nb_sp_minus - TNp, nb_su_plus - FNu, nb_su_minus - TNu]
    
    t = problem.variables.type

    problem.variables.add(obj = objective,
                        lb = lower_bounds,
                        ub = upper_bounds,
                        names = names,
                        types = [t.integer, t.integer, t.integer, t.integer]) # [t.continuous, t.binary, t.integer]
        


    # Constraints
    Up_c = U - nb_sp_minus - nb_su_minus
    Low_c = L - nb_sp_minus - nb_su_minus

    acc_constraint = [["sp_plus", "sp_minus", "su_plus", "su_minus"], [1.0, -1.0, 1.0, -1.0]]
    
    if fairnessMetric == 1:
        tot_p = (nb_sp_plus + nb_sp_minus)
        tot_u = (nb_su_plus + nb_su_minus)
        const_fairness = fairness_tolerence * tot_p * tot_u
        unf_constraint = [["sp_plus", "sp_minus", "su_plus", "su_minus"], [tot_u, tot_u, -tot_p, -tot_p]]
        constraints = [ acc_constraint, acc_constraint, unf_constraint, unf_constraint ]
        constraint_names = ["acc_l", "acc_u", "unf_l", "unf_u"]
        rhs = [Low_c, Up_c, -const_fairness, const_fairness]
        constraint_senses = [ "G", "L", "G", "L"]
    elif fairnessMetric == 3:
        const_fairness = fairness_tolerence * nb_su_minus * nb_sp_minus
        unf_constraint = [["sp_minus", "su_minus"], [nb_su_minus, -nb_sp_minus]]
        constraints = [ acc_constraint, acc_constraint, unf_constraint, unf_constraint ]
        constraint_names = ["acc_l", "acc_u", "unf_l", "unf_u"]
        rhs = [Low_c, Up_c, -const_fairness, const_fairness]
        constraint_senses = [ "G", "L", "G", "L"]
    elif fairnessMetric == 4:
        const_fairness = fairness_tolerence * nb_su_plus * nb_sp_plus
        unf_constraint = [["sp_plus", "su_plus"], [nb_su_plus, -nb_sp_plus]]
        constraints = [ acc_constraint, acc_constraint, unf_constraint, unf_constraint ]
        constraint_names = ["acc_l", "acc_u", "unf_l", "unf_u"]
        rhs = [Low_c, Up_c, -const_fairness, const_fairness]
        constraint_senses = [ "G", "L", "G", "L"]
    elif fairnessMetric == 5:
        const_fairness_3 = fairness_tolerence * nb_su_minus * nb_sp_minus
        unf_constraint_3 = [["sp_minus", "su_minus"], [nb_su_minus, -nb_sp_minus]]
        const_fairness_4 = fairness_tolerence * nb_su_plus * nb_sp_plus
        unf_constraint_4 = [["sp_plus", "su_plus"], [nb_su_plus, -nb_sp_plus]]
        constraints = [ acc_constraint, acc_constraint, unf_constraint_3, unf_constraint_3, unf_constraint_4, unf_constraint_4 ]
        constraint_names = ["acc_l", "acc_u", "unf_l_3", "unf_u_3", "unf_l_4", "unf_u_4"]
        rhs = [Low_c, Up_c, -const_fairness_3, const_fairness_3, -const_fairness_4, const_fairness_4]
        constraint_senses = [ "G", "L", "G", "L", "G", "L"]
    else:
        print("Unknown metric ", fairnessMetric, ". Exiting.")
        exit()

    # And add the constraints
    problem.linear_constraints.add(lin_expr = constraints,
                                senses = constraint_senses,
                                rhs = rhs,
                                names = constraint_names)

    #ps = problem.create_parameter_set()
    #ps.add(problem.parameters.threads, 1) #c.parameters.threads.values.1)
    problem.parameters.threads.set(1)
    #print("current val = ", problem.parameters.threads.get())
    # Solve the problem
    status = problem.solve()
    status = problem.solution.get_status()
    
    #print(status)
    if status == 101: # OPT
        #print(problem.solution.get_values())
        #print(problem.solution.get_objective_value() + nb_sp_minus + nb_su_minus)
        nb_well_classified = problem.solution.get_objective_value() + nb_sp_minus + nb_su_minus
        if check_response:
            if nb_well_classified > U or nb_well_classified < L:
                print("Invalid solution found: # well classified = ", nb_well_classified, ", L = ", L, ", U = ", U)
                exit()
            sp_plus_val, sp_minus_val, su_plus_val, su_minus_val = problem.solution.get_values()
            if fairnessMetric == 1:
                unfairness = abs( ((sp_plus_val+sp_minus_val)/(nb_sp_plus+nb_sp_minus)) - ((su_plus_val+su_minus_val)/(nb_su_plus+nb_su_minus)) )
            elif fairnessMetric == 3:
                unfairness = abs( (sp_minus/nb_sp_minus) - (su_minus/nb_su_minus) )
            elif fairnessMetric == 4:
                unfairness = abs( (sp_plus/nb_sp_plus) - (su_plus/nb_su_plus) )
            elif fairnessMetric == 5:
                unfairness3 = abs( (sp_minus/nb_sp_minus) - (su_minus/nb_su_minus) )
                unfairness4 = abs( (sp_plus/nb_sp_plus) - (su_plus/nb_su_plus) )
                unfairness = max(unfairness3, unfairness4)
            if unfairness > fairness_tolerence:
                print("Invalid solution found: Unfairness (metric %d) is %f > epsilon = %f" %(fairnessMetric, unfairness, fairness_tolerence))
                exit()
        return nb_well_classified
    elif status == 103:
        #print("Infeasible problem!")
        return -1
    else:
        print("Unknown status: ", status, ". Exiting!")
        exit()