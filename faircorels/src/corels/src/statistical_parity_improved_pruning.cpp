#include <mistral_solver.hpp>
#include <mistral_variable.hpp>
#include <mistral_search.hpp>

using namespace std;

class FilteringStatisticalParity {

private:

	Mistral::VarArray scope;
	Mistral::Solver s;
	Mistral::Outcome result;
	//
	int nb_protected, nb_unprotected, nb_protected_negative, nb_unprotected_negative ;

	//L is a lower bound for the accuracy
	int L ;

	//U is a lower bound for the accuracy
	int U ;
	//fairness_tolerence is a float in [0,1] that represents the unfairness tolerated.
	//If this value is 0 then the model is 100% fair.
	float fairness_tolerence ;

public:
	/*	ub_sp_plus is the number of examples protected with a positive class
		ub_sp_minus is the number of examples protected with a negative class
		ub_su_plus is the number of examples unprotected with a positive class
		ub_su_minus is the number of examples unprotected with a negative class
		TPp is the number of True Positive protected instances (among instances captured by the prefix)
	 */
	FilteringStatisticalParity(
			int nb_sp_plus,
			int nb_sp_minus,
			int nb_su_plus,
			int nb_su_minus,
			int L,
			int U,
			float fairness_tolerence,
			int TPp,
			int FPp,
			int TNp,
			int FNp,
			int TPu,
			int FPu,
			int TNu,
			int FNu
			 ) :
				nb_protected( nb_sp_plus+ nb_sp_minus),
				nb_unprotected( nb_su_plus+ nb_su_minus),
				nb_protected_negative (nb_sp_minus),
				nb_unprotected_negative ( nb_su_minus),
				L(L),
				U(U),
				fairness_tolerence(fairness_tolerence)
{

		//scope[0] is sp_plus
		//scope.add(Variable(0,nb_sp_plus) );
		scope.add(Mistral::Variable(TPp, nb_sp_plus - FNp) );
		//scope[1] is sp_minus
		scope.add(Mistral::Variable(FPp, nb_sp_minus - TNp) );
		//scope[2] is su_plus
		scope.add(Mistral::Variable(TPu, nb_su_plus - FNu) );
		//scope[3] is su_minus
		scope.add(Mistral::Variable(FPu, nb_su_minus - TNu) );

		result =UNSAT;

		int constant ;

		std::vector<int> accuracy_coefficient ;
		// accuracy_coefficient represents the coefficients used in the accuracy.
		accuracy_coefficient.push_back(1);
		accuracy_coefficient.push_back(-1);
		accuracy_coefficient.push_back(1);
		accuracy_coefficient.push_back(-1);

		//Accuracy constraints
		constant  = U  -  nb_sp_minus - nb_su_minus ;
		//std::cout <<  " \n c in the CP model, the upper Bound for accuracy constraint: "  << constant	<< std::endl;
		s.add( Sum(scope, accuracy_coefficient) <= constant  ) ;

		constant  = L  -   nb_sp_minus - nb_su_minus ;

		//std::cout <<  " c in the CP model, the lower Bound for accuracy constraint: "  << constant	<< std::endl;
		s.add( Sum(scope, accuracy_coefficient) >= constant  ) ;

		//Fairness constraints
		std::vector<int> fairness_coefficient;


		fairness_coefficient.push_back(nb_unprotected);
		fairness_coefficient.push_back(nb_unprotected);


		fairness_coefficient.push_back(-nb_protected);
		fairness_coefficient.push_back(-nb_protected);

		int fairness_bound = (int ) ( (float) nb_protected *(float) nb_unprotected * fairness_tolerence );

		//std::cout <<  " c fairness_bound : "  << fairness_bound	<< std::endl;
		//Variable bound_fairness (-fairness_bound, fairness_bound);

		s.add( Sum(scope, fairness_coefficient) <= fairness_bound ) ;
		s.add( Sum(scope, fairness_coefficient) >= (-fairness_bound) ) ;

		//An alternative way:
		//fairness_coefficient.push_back(-1);
		//scope.add(bound_fairness);
		//s.add( Sum(scope, fairness_coefficient, 0, 0) );
		//s.add( Sum(scope, fairness_coefficient) == bound_fairness) ;
}



	void run(int verbosity){
		s.parameters.verbosity = verbosity;
		//s.parameters.time_limit = 0.2; // TODO add arg
		//std::cout <<  s << std::endl;
		//s.rewrite();
		s.consolidate();
		if(L > U){
			printf("Lower and upper bound meet. : L = %d ; U = %d\n", L, U);
			result = UNSAT;
		} else {
			result =  s.solve();
		}
	}

	void print_statistics(){
		s.statistics.print_full(std::cout);
	}

    bool isFeasible(){
        if (result )
		{
            return true;
        } else {
            return false;
        }
    }

	void print_and_verify_solution(){

		if (result )
		{
			std::cout <<  " \n \n c Solution Found! "  << 	std::endl;

			//for ( unsigned int i= 0 ; i< scope.size ; ++i)
			//	std::cout <<  " c Solution value of scope[" << i <<  "]  is " << scope[i].get_solution_int_value() << std::endl;


			int sp_plus= scope[0].get_solution_int_value();
			int sp_minus= scope[1].get_solution_int_value();
			int su_plus= scope[2].get_solution_int_value();
			int su_minus= scope[3].get_solution_int_value();

			//int fairness_sum= scope[4].get_solution_int_value();

			std::cout <<  " c nb protected positive that are predicted positive: " <<  sp_plus << std::endl;
			std::cout <<  " c nb protected negative that are predicted positive: " <<  sp_minus << std::endl;
			std::cout <<  " c nb unprotected positive that are predicted positive: " <<  su_plus << std::endl;
			std::cout <<  " c nb unprotected negative that are predicted positive:  " <<  su_minus << std::endl;
			//std::cout <<  " c nb fairness_sum  " <<  fairness_sum << std::endl;
			printf("nb_protected_negative : %d\n", nb_protected_negative);
			printf("nb_unprotected_negative : %d\n", nb_unprotected_negative);
			int accuracy = sp_plus + su_plus -sp_minus - su_minus + nb_protected_negative + nb_unprotected_negative;

			float fairness = ((float)(sp_plus+sp_minus) /(float) nb_protected )  - 	 ((float)(su_plus +su_minus) /(float) nb_unprotected );

			std::cout <<  " c Accuracy ( Number of examamples well classified) is "  << accuracy  << " out of " << nb_protected+ nb_unprotected << 	std::endl;

			std::cout <<  " c Fairness (as float) is "  << fairness << std::endl;

			assert (accuracy >= L   ) ;
			assert (accuracy <= U   ) ;
			assert (fairness <= fairness_tolerence    ) ;
			assert (fairness >= - fairness_tolerence    ) ;
			//int discrete_fairness= (nb_unprotected * (sp_plus+sp_minus)) - (nb_protected * (su_plus +su_minus) )  ;
			//std::cout <<  " c Fairness (discrete constraint) is "  << discrete_fairness << std::endl;
			std::cout <<  " c Solution Verified"  << 	std::endl;

		}
		else
			std::cout <<  " c No Solution! "  << 	std::endl;
	}


};

class FilteringPredictiveEquality {

private:

	Mistral::VarArray scope;
	Mistral::Solver s;
	Mistral::Outcome result;
	//
	int nb_protected, nb_unprotected, nb_protected_negative, nb_unprotected_negative ;

	//L is a lower bound for the accuracy
	int L ;

	//U is a upper bound for the accuracy
	int U;
	//fairness_tolerence is a float in [0,1] that represents the unfairness tolerated.
	//If this value is 0 then the model is 100% fair.
	float fairness_tolerence ;


public:
	/*	ub_sp_plus is the number of examples protected with a positive class
		ub_sp_minus is the number of examples protected with a negative class
		ub_su_plus is the number of examples unprotected with a positive class
		ub_su_minus is the number of examples unprotected with a negative class
		TPp is the number of True Positive protected instances (among instances captured by the prefix)
	 */
	FilteringPredictiveEquality(
			int nb_sp_plus,
			int nb_sp_minus,
			int nb_su_plus,
			int nb_su_minus,
			int L,
			int U,
			float fairness_tolerence,
			int TPp,
			int FPp,
			int TNp,
			int FNp,
			int TPu,
			int FPu,
			int TNu,
			int FNu
			 ) :
				nb_protected( nb_sp_plus+ nb_sp_minus),
				nb_unprotected( nb_su_plus+ nb_su_minus),
				nb_protected_negative (nb_sp_minus),
				nb_unprotected_negative ( nb_su_minus),
				L(L),
				U(U),
				fairness_tolerence(fairness_tolerence)
{

		//scope[0] is sp_plus
		//scope.add(Variable(0,nb_sp_plus) );
		scope.add(Mistral::Variable(TPp, nb_sp_plus - FNp) );
		//scope[1] is sp_minus
		scope.add(Mistral::Variable(FPp, nb_sp_minus - TNp) );
		//scope[2] is su_plus
		scope.add(Mistral::Variable(TPu, nb_su_plus - FNu) );
		//scope[3] is su_minus
		scope.add(Mistral::Variable(FPu, nb_su_minus - TNu) );

		result =UNSAT;

		int constant ;

		std::vector<int> accuracy_coefficient ;
		// accuracy_coefficient represents the coefficients used in the accuracy.
		accuracy_coefficient.push_back(1);
		accuracy_coefficient.push_back(-1);
		accuracy_coefficient.push_back(1);
		accuracy_coefficient.push_back(-1);

		//Accuracy constraints
		constant  = U  -  nb_sp_minus - nb_su_minus ;
		//std::cout <<  " \n c in the CP model, the upper Bound for accuracy constraint: "  << constant	<< std::endl;
		s.add( Sum(scope, accuracy_coefficient) <= constant  ) ;

		constant  = L  -   nb_sp_minus - nb_su_minus ;

		//std::cout <<  " c in the CP model, the lower Bound for accuracy constraint: "  << constant	<< std::endl;
		s.add( Sum(scope, accuracy_coefficient) >= constant  ) ;

		//Fairness constraints
		std::vector<int> fairness_coefficient;

		fairness_coefficient.push_back(0);
		fairness_coefficient.push_back(nb_unprotected_negative);


		fairness_coefficient.push_back(0);
		fairness_coefficient.push_back(-nb_protected_negative);

		int fairness_bound = (int ) ( (float) nb_unprotected_negative *(float) nb_protected_negative * fairness_tolerence );

		//std::cout <<  " c fairness_bound : "  << fairness_bound	<< std::endl;
		//Variable bound_fairness (-fairness_bound, fairness_bound);

		s.add( Sum(scope, fairness_coefficient) <= fairness_bound ) ;
		s.add( Sum(scope, fairness_coefficient) >= (-fairness_bound) ) ;

		//An alternative way:
		//fairness_coefficient.push_back(-1);
		//scope.add(bound_fairness);
		//s.add( Sum(scope, fairness_coefficient, 0, 0) );
		//s.add( Sum(scope, fairness_coefficient) == bound_fairness) ;
}



	void run(int verbosity){
		s.parameters.verbosity = verbosity;
		//s.parameters.time_limit = 0.2; // TODO add arg
		//std::cout <<  s << std::endl;
		//s.rewrite();
		s.consolidate();
		if(L > U){
			printf("Lower and upper bound meet. : L = %d ; U = %d\n", L, U);
			result = UNSAT;
		} else {
			result =  s.solve();
		}
	}

	void print_statistics(){
		s.statistics.print_full(std::cout);
	}

    bool isFeasible(){
        if (result )
		{
            return true;
        } else {
            return false;
        }
    }


	void print_and_verify_solution(){

		if (result == SAT)
		{
			std::cout <<  " \n \n c Solution Found! "  << 	std::endl;

			//for ( unsigned int i= 0 ; i< scope.size ; ++i)
			//	std::cout <<  " c Solution value of scope[" << i <<  "]  is " << scope[i].get_solution_int_value() << std::endl;


			int sp_plus= scope[0].get_solution_int_value();
			int sp_minus= scope[1].get_solution_int_value();
			int su_plus= scope[2].get_solution_int_value();
			int su_minus= scope[3].get_solution_int_value();

			//int fairness_sum= scope[4].get_solution_int_value();

			std::cout <<  " c nb protected positive that are predicted positive: " <<  sp_plus << std::endl;
			std::cout <<  " c nb protected negative that are predicted positive: " <<  sp_minus << std::endl;
			std::cout <<  " c nb unprotected positive that are predicted positive: " <<  su_plus << std::endl;
			std::cout <<  " c nb unprotected negative that are predicted positive:  " <<  su_minus << std::endl;
			//std::cout <<  " c nb fairness_sum  " <<  fairness_sum << std::endl;
			printf("nb_protected_negative : %d\n", nb_protected_negative);
			printf("nb_unprotected_negative : %d\n", nb_unprotected_negative);
			int accuracy = sp_plus + su_plus -sp_minus - su_minus + nb_protected_negative + nb_unprotected_negative;

			float fairness = ((float)sp_minus/(float) (sp_minus+(nb_protected_negative-sp_minus))  )  - 	 ((float)su_minus/(float) (su_minus+(nb_unprotected_negative-su_minus))  );

			std::cout <<  " c Accuracy ( Number of examamples well classified) is "  << accuracy  << " out of " << nb_protected+ nb_unprotected << 	std::endl;

			std::cout <<  " c PP Unfairness (as float) is "  << fairness << std::endl;

			assert (accuracy >= L   ) ;
			assert (accuracy <= U   ) ;
			assert (fairness <= fairness_tolerence    ) ;
			assert (fairness >= - fairness_tolerence    ) ;
			//int discrete_fairness= (nb_unprotected * (sp_plus+sp_minus)) - (nb_protected * (su_plus +su_minus) )  ;
			//std::cout <<  " c Fairness (discrete constraint) is "  << discrete_fairness << std::endl;
			std::cout <<  " c Solution Verified"  << 	std::endl;

		}
		else if (result == UNSAT)
			std::cout <<  " c No Solution! "  << 	std::endl;
		else if (result == LIMITOUT)
			std::cout <<  " c Timeout reached! "  << 	std::endl;
		else
			std::cout <<  " c Other status! "  << 	std::endl;
	}


};

class FilteringPredictiveParity {

private:

	Mistral::VarArray scope;
	Mistral::Solver s;
	Mistral::Outcome result;
	//
	int nb_protected, nb_unprotected, nb_protected_negative, nb_unprotected_negative ;

	//L is a lower bound for the accuracy
	int L ;

	//U is a upper bound for the accuracy
	int U;
	//fairness_tolerence is a float in [0,1] that represents the unfairness tolerated.
	//If this value is 0 then the model is 100% fair.
	float fairness_tolerence ;


public:
	/*	ub_sp_plus is the number of examples protected with a positive class
		ub_sp_minus is the number of examples protected with a negative class
		ub_su_plus is the number of examples unprotected with a positive class
		ub_su_minus is the number of examples unprotected with a negative class
		TPp is the number of True Positive protected instances (among instances captured by the prefix)
	 */
	FilteringPredictiveParity(
			int nb_sp_plus,
			int nb_sp_minus,
			int nb_su_plus,
			int nb_su_minus,
			int L,
			int U,
			float fairness_tolerence,
			int TPp,
			int FPp,
			int TNp,
			int FNp,
			int TPu,
			int FPu,
			int TNu,
			int FNu
			 ) :
				nb_protected( nb_sp_plus+ nb_sp_minus),
				nb_unprotected( nb_su_plus+ nb_su_minus),
				nb_protected_negative (nb_sp_minus),
				nb_unprotected_negative ( nb_su_minus),
				L(L),
				U(U),
				fairness_tolerence(fairness_tolerence)
{

		//scope[0] is sp_plus
		scope.add(Mistral::Variable(TPp, nb_sp_plus - FNp) );
		//scope[1] is sp_minus
		scope.add(Mistral::Variable(FPp, nb_sp_minus - TNp) );
		//scope[2] is su_plus
		scope.add(Mistral::Variable(TPu, nb_su_plus - FNu) );
		//scope[3] is su_minus
		scope.add(Mistral::Variable(FPu, nb_su_minus - TNu) );

		int reductionFactor = 1000; // might introduce inexactitudes but is necessary so that the variables are int without overflow...
		//scope[4] is sp_plus*su_plus
		scope.add(Mistral::Variable((TPp*TPu/reductionFactor), (nb_sp_plus - FNp)*(nb_su_plus - FNu)/reductionFactor) );
		//scope[5] is sp_plus*su_minus
		scope.add(Mistral::Variable((TPp*FPu/reductionFactor), (nb_sp_plus - FNp)*(nb_su_minus - TNu)/reductionFactor) );
		//scope[6] is sp_minus*su_plus
		scope.add(Mistral::Variable((FPp*TPu/reductionFactor), (nb_sp_minus - TNp)*(nb_su_plus - FNu)/reductionFactor) );
		//scope[7] is sp_minus*su_minus
		scope.add(Mistral::Variable((FPp*FPu/reductionFactor), (nb_sp_minus - TNp)*(nb_su_minus - TNu)/reductionFactor) );

		s.add(scope[0] * scope[2]  / reductionFactor == scope[4]);
		s.add(scope[0] * scope[3]  / reductionFactor == scope[5]);
		s.add(scope[1] * scope[2]  / reductionFactor == scope[6]);
		s.add(scope[1] * scope[3] / reductionFactor == scope[7]);

		result =UNSAT;

		int constant ;

		std::vector<int> accuracy_coefficient ;
		// accuracy_coefficient represents the coefficients used in the accuracy.
		accuracy_coefficient.push_back(1);
		accuracy_coefficient.push_back(-1);
		accuracy_coefficient.push_back(1);
		accuracy_coefficient.push_back(-1);
		accuracy_coefficient.push_back(0);
		accuracy_coefficient.push_back(0);
		accuracy_coefficient.push_back(0);
		accuracy_coefficient.push_back(0);

		//Accuracy constraints
		constant  = U  -  nb_sp_minus - nb_su_minus ;
		//std::cout <<  " \n c in the CP model, the upper Bound for accuracy constraint: "  << constant	<< std::endl;
		s.add( Sum(scope, accuracy_coefficient) <= constant  ) ;

		constant  = L  -   nb_sp_minus - nb_su_minus ;

		//std::cout <<  " c in the CP model, the lower Bound for accuracy constraint: "  << constant	<< std::endl;
		s.add( Sum(scope, accuracy_coefficient) >= constant  ) ;

		//Fairness constraints
		// I need int coeffs. Here I assume fairness_tolerence is in [0,0.5] (reasonable) and has two decimals or less (if more decimals, need change)
		int intFactor = 100;

		std::vector<int> fairness_coefficients1;

		fairness_coefficients1.push_back(0);
		fairness_coefficients1.push_back(0);
		fairness_coefficients1.push_back(0);
		fairness_coefficients1.push_back(0);
		fairness_coefficients1.push_back(intFactor*(fairness_tolerence));
		fairness_coefficients1.push_back(intFactor*(fairness_tolerence-1));
		fairness_coefficients1.push_back(intFactor*(fairness_tolerence+1));
		fairness_coefficients1.push_back(intFactor*(fairness_tolerence));

		s.add( Sum(scope, fairness_coefficients1) > 0  ) ; // strictly

		std::vector<int> fairness_coefficients2;

		fairness_coefficients2.push_back(0);
		fairness_coefficients2.push_back(0);
		fairness_coefficients2.push_back(0);
		fairness_coefficients2.push_back(0);
		fairness_coefficients2.push_back(intFactor*(-fairness_tolerence));
		fairness_coefficients2.push_back(intFactor*(-fairness_tolerence-1));
		fairness_coefficients2.push_back(intFactor*(-fairness_tolerence+1));
		fairness_coefficients2.push_back(intFactor*(-fairness_tolerence));

		s.add( Sum(scope, fairness_coefficients2) < 0  ) ; // strictly
}



	void run(int verbosity){
		s.parameters.verbosity = verbosity;
		//s.parameters.time_limit = 0.2; // TODO add arg
		//std::cout <<  s << std::endl;
		//s.rewrite();
		s.consolidate();
		if(L > U){
			printf("Lower and upper bound meet. : L = %d ; U = %d\n", L, U);
			result = UNSAT;
		} else {
			result =  s.solve();
		}
	}

	void print_statistics(){
		s.statistics.print_full(std::cout);
	}


	bool isFeasible(){
        if (result )
		{
            return true;
        } else {
            return false;
        }
    }

	void print_and_verify_solution(){

		if (result )
		{
			std::cout <<  " \n \n c Solution Found! "  << 	std::endl;

			//for ( unsigned int i= 0 ; i< scope.size ; ++i)
			//	std::cout <<  " c Solution value of scope[" << i <<  "]  is " << scope[i].get_solution_int_value() << std::endl;


			int sp_plus= scope[0].get_solution_int_value();
			int sp_minus= scope[1].get_solution_int_value();
			int su_plus= scope[2].get_solution_int_value();
			int su_minus= scope[3].get_solution_int_value();

			//int fairness_sum= scope[4].get_solution_int_value();

			std::cout <<  " c nb protected positive that are predicted positive: " <<  sp_plus << std::endl;
			std::cout <<  " c nb protected negative that are predicted positive: " <<  sp_minus << std::endl;
			std::cout <<  " c nb unprotected positive that are predicted positive: " <<  su_plus << std::endl;
			std::cout <<  " c nb unprotected negative that are predicted positive:  " <<  su_minus << std::endl;
			//std::cout <<  " c nb fairness_sum  " <<  fairness_sum << std::endl;
			printf("nb_protected_negative : %d\n", nb_protected_negative);
			printf("nb_unprotected_negative : %d\n", nb_unprotected_negative);
			int accuracy = sp_plus + su_plus -sp_minus - su_minus + nb_protected_negative + nb_unprotected_negative;

			float fairness = ((float)sp_plus /(float) (sp_plus+sp_minus) )  - 	 ((float)su_plus /(float) (su_plus +su_minus)) ; // fmax(1.0,...) pas nécessaire grâce aux '<' et '>'

			std::cout <<  " c Accuracy ( Number of examamples well classified) is "  << accuracy  << " out of " << nb_protected+ nb_unprotected << 	std::endl;

			std::cout <<  " c Fairness (as float) is "  << fairness << std::endl;

			assert (accuracy >= L   ) ;
			assert (accuracy <= U   ) ;
			assert (fairness <= fairness_tolerence    ) ;
			printf("Fairness = %.6f ; - fairness_tolerence = %.6f\n", fairness, -fairness_tolerence);
			assert (fairness >= - fairness_tolerence    ) ;
			//int discrete_fairness= (nb_unprotected * (sp_plus+sp_minus)) - (nb_protected * (su_plus +su_minus) )  ;
			//std::cout <<  " c Fairness (discrete constraint) is "  << discrete_fairness << std::endl;
			std::cout <<  " c Solution Verified"  << 	std::endl;

		}
		else
			std::cout <<  " c No Solution! "  << 	std::endl;
	}


};

class FilteringEqualOpportunity {
	
private:

	Mistral::VarArray scope;
	Mistral::Solver s;
	Mistral::Outcome result;
	//
	int nb_protected, nb_unprotected, nb_protected_negative, nb_unprotected_negative ;

	//L is a lower bound for the accuracy
	int L ;

	//U is a upper bound for the accuracy
	int U;
	//fairness_tolerence is a float in [0,1] that represents the unfairness tolerated.
	//If this value is 0 then the model is 100% fair.
	float fairness_tolerence ;


public:
	/*	ub_sp_plus is the number of examples protected with a positive class
		ub_sp_minus is the number of examples protected with a negative class
		ub_su_plus is the number of examples unprotected with a positive class
		ub_su_minus is the number of examples unprotected with a negative class
		TPp is the number of True Positive protected instances (among instances captured by the prefix)
	 */
	FilteringEqualOpportunity(
			int nb_sp_plus,
			int nb_sp_minus,
			int nb_su_plus,
			int nb_su_minus,
			int L,
			int U,
			float fairness_tolerence,
			int TPp,
			int FPp,
			int TNp,
			int FNp,
			int TPu,
			int FPu,
			int TNu,
			int FNu
			 ) :
				nb_protected( nb_sp_plus+ nb_sp_minus),
				nb_unprotected( nb_su_plus+ nb_su_minus),
				nb_protected_negative (nb_sp_minus),
				nb_unprotected_negative ( nb_su_minus),
				L(L),
				U(U),
				fairness_tolerence(fairness_tolerence)
{

		//scope[0] is sp_plus
		//scope.add(Variable(0,nb_sp_plus) );
		scope.add(Mistral::Variable(TPp, nb_sp_plus - FNp) );
		//scope[1] is sp_minus
		scope.add(Mistral::Variable(FPp, nb_sp_minus - TNp) );
		//scope[2] is su_plus
		scope.add(Mistral::Variable(TPu, nb_su_plus - FNu) );
		//scope[3] is su_minus
		scope.add(Mistral::Variable(FPu, nb_su_minus - TNu) );
		
		result =UNSAT;

		int constant ;

		std::vector<int> accuracy_coefficient ;
		// accuracy_coefficient represents the coefficients used in the accuracy.
		accuracy_coefficient.push_back(1);
		accuracy_coefficient.push_back(-1);
		accuracy_coefficient.push_back(1);
		accuracy_coefficient.push_back(-1);

		//Accuracy constraints
		constant  = U  -  nb_sp_minus - nb_su_minus ;
		//std::cout <<  " \n c in the CP model, the upper Bound for accuracy constraint: "  << constant	<< std::endl;
		s.add( Sum(scope, accuracy_coefficient) <= constant  ) ;

		constant  = L  -   nb_sp_minus - nb_su_minus ;

		//std::cout <<  " c in the CP model, the lower Bound for accuracy constraint: "  << constant	<< std::endl;
		s.add( Sum(scope, accuracy_coefficient) >= constant  ) ;

		//Fairness constraints
		std::vector<int> fairness_coefficient;

		fairness_coefficient.push_back(-nb_su_plus);
		fairness_coefficient.push_back(0);


		fairness_coefficient.push_back(nb_sp_plus);
		fairness_coefficient.push_back(0);

		int fairness_bound = (int ) ( (float) nb_su_plus *(float) nb_sp_plus * fairness_tolerence );

		//std::cout <<  " c fairness_bound : "  << fairness_bound	<< std::endl;
		//Variable bound_fairness (-fairness_bound, fairness_bound);

		s.add( Sum(scope, fairness_coefficient) <= fairness_bound ) ;
		s.add( Sum(scope, fairness_coefficient) >= (-fairness_bound) ) ;

		//An alternative way:
		//fairness_coefficient.push_back(-1);
		//scope.add(bound_fairness);
		//s.add( Sum(scope, fairness_coefficient, 0, 0) );
		//s.add( Sum(scope, fairness_coefficient) == bound_fairness) ;
}



	void run(int verbosity){
		s.parameters.verbosity = verbosity;
		//s.parameters.time_limit = 0.2; // TODO add arg
		//std::cout <<  s << std::endl;
		//s.rewrite();
		s.consolidate();
		if(L > U){
			printf("Lower and upper bound meet. : L = %d ; U = %d\n", L, U);
			result = UNSAT;
		} else {
			result =  s.solve();
		}
	}

	void print_statistics(){
		s.statistics.print_full(std::cout);
	}


	bool isFeasible(){
        if (result )
		{
            return true;
        } else {
            return false;
        }
    }


	void print_and_verify_solution(){

		if (result == SAT)
		{
			std::cout <<  " \n \n c Solution Found! "  << 	std::endl;

			//for ( unsigned int i= 0 ; i< scope.size ; ++i)
			//	std::cout <<  " c Solution value of scope[" << i <<  "]  is " << scope[i].get_solution_int_value() << std::endl;


			int sp_plus= scope[0].get_solution_int_value();
			int sp_minus= scope[1].get_solution_int_value();
			int su_plus= scope[2].get_solution_int_value();
			int su_minus= scope[3].get_solution_int_value();

			//int fairness_sum= scope[4].get_solution_int_value();

			std::cout <<  " c nb protected positive that are predicted positive: " <<  sp_plus << std::endl;
			std::cout <<  " c nb protected negative that are predicted positive: " <<  sp_minus << std::endl;
			std::cout <<  " c nb unprotected positive that are predicted positive: " <<  su_plus << std::endl;
			std::cout <<  " c nb unprotected negative that are predicted positive:  " <<  su_minus << std::endl;
			//std::cout <<  " c nb fairness_sum  " <<  fairness_sum << std::endl;
			//printf("nb_protected_negative : %d\n", nb_protected_negative);
			//printf("nb_unprotected_negative : %d\n", nb_unprotected_negative);
			int accuracy = sp_plus + su_plus -sp_minus - su_minus + nb_protected_negative + nb_unprotected_negative;

			float fairness = ((float)((nb_protected - nb_protected_negative - sp_plus) /(float) (nb_protected - nb_protected_negative)  )  - 	 ((float)(nb_unprotected - nb_unprotected_negative - su_plus)/(float) (nb_unprotected - nb_unprotected_negative)  ));

			std::cout <<  " c Accuracy ( Number of examamples well classified) is "  << accuracy  << " out of " << nb_protected+ nb_unprotected << 	std::endl;

			std::cout <<  " c EO Unfairness (as float) is "  << fairness << std::endl;
			std::cout <<  " c EO Unfairness tolerance (as float) is "  << fairness_tolerence << std::endl;
			assert (accuracy >= L   ) ;
			assert (accuracy <= U   ) ;
			assert (fabs(fairness - fairness_tolerence) <=  10E-6   ) ;
			//int discrete_fairness= (nb_unprotected * (sp_plus+sp_minus)) - (nb_protected * (su_plus +su_minus) )  ;
			//std::cout <<  " c Fairness (discrete constraint) is "  << discrete_fairness << std::endl;
			std::cout <<  " c Solution Verified"  << 	std::endl;

		}
		else if (result == UNSAT)
			std::cout <<  " c No Solution! "  << 	std::endl;
		else if (result == LIMITOUT)
			std::cout <<  " c Timeout reached! "  << 	std::endl;
		else
			std::cout <<  " c Other status! "  << 	std::endl;
	}


};

class FilteringEqualizedOdds {

private:

	Mistral::VarArray scope;
	Mistral::Solver s;
	Mistral::Outcome result;
	//
	int nb_protected, nb_unprotected, nb_protected_negative, nb_unprotected_negative ;

	//L is a lower bound for the accuracy
	int L ;

	//U is a upper bound for the accuracy
	int U;
	//fairness_tolerence is a float in [0,1] that represents the unfairness tolerated.
	//If this value is 0 then the model is 100% fair.
	float fairness_tolerence ;


public:
	/*	ub_sp_plus is the number of examples protected with a positive class
		ub_sp_minus is the number of examples protected with a negative class
		ub_su_plus is the number of examples unprotected with a positive class
		ub_su_minus is the number of examples unprotected with a negative class
		TPp is the number of True Positive protected instances (among instances captured by the prefix)
	 */
	FilteringEqualizedOdds(
			int nb_sp_plus,
			int nb_sp_minus,
			int nb_su_plus,
			int nb_su_minus,
			int L,
			int U,
			float fairness_tolerence,
			int TPp,
			int FPp,
			int TNp,
			int FNp,
			int TPu,
			int FPu,
			int TNu,
			int FNu
			 ) :
				nb_protected( nb_sp_plus+ nb_sp_minus),
				nb_unprotected( nb_su_plus+ nb_su_minus),
				nb_protected_negative (nb_sp_minus),
				nb_unprotected_negative ( nb_su_minus),
				L(L),
				U(U),
				fairness_tolerence(fairness_tolerence)
{

		//scope[0] is sp_plus
		//scope.add(Variable(0,nb_sp_plus) );
		scope.add(Mistral::Variable(TPp, nb_sp_plus - FNp) );
		//scope[1] is sp_minus
		scope.add(Mistral::Variable(FPp, nb_sp_minus - TNp) );
		//scope[2] is su_plus
		scope.add(Mistral::Variable(TPu, nb_su_plus - FNu) );
		//scope[3] is su_minus
		scope.add(Mistral::Variable(FPu, nb_su_minus - TNu) );
		
		result =UNSAT;

		int constant ;

		std::vector<int> accuracy_coefficient ;
		// accuracy_coefficient represents the coefficients used in the accuracy.
		accuracy_coefficient.push_back(1);
		accuracy_coefficient.push_back(-1);
		accuracy_coefficient.push_back(1);
		accuracy_coefficient.push_back(-1);

		//Accuracy constraints
		constant  = U  -  nb_sp_minus - nb_su_minus ;
		//std::cout <<  " \n c in the CP model, the upper Bound for accuracy constraint: "  << constant	<< std::endl;
		s.add( Sum(scope, accuracy_coefficient) <= constant  ) ;

		constant  = L  -   nb_sp_minus - nb_su_minus ;

		//std::cout <<  " c in the CP model, the lower Bound for accuracy constraint: "  << constant	<< std::endl;
		s.add( Sum(scope, accuracy_coefficient) >= constant  ) ;

		//Fairness constraints ------------------------------------------
		std::vector<int> fairness_coefficient1;

		fairness_coefficient1.push_back((nb_su_minus*nb_sp_minus)/nb_sp_plus);
		fairness_coefficient1.push_back(0);


		fairness_coefficient1.push_back(-(nb_su_minus*nb_sp_minus)/nb_su_plus);
		fairness_coefficient1.push_back(0);

		std::vector<int> fairness_coefficient2;

		fairness_coefficient2.push_back(0);
		fairness_coefficient2.push_back(nb_su_minus);


		fairness_coefficient2.push_back(0);
		fairness_coefficient2.push_back(-nb_sp_minus);

		int fairness_bound = (int ) ( (float) nb_su_minus *(float) nb_sp_minus * fairness_tolerence );
		//printf("fairness bound = %d\n", fairness_bound);
		//int my_number = 2147483647;
		//printf("my number is big %d (should be 2147483647)\n", my_number);
		//int my_other_number = 2147483648;
		//printf("my other number is too big : %d\n", my_other_number);
		//std::cout <<  " c fairness_bound : "  << fairness_bound	<< std::endl;
		//Variable bound_fairness (-fairness_bound, fairness_bound);

		s.add(Abs(Sum(scope, fairness_coefficient1)) + Abs(Sum(scope, fairness_coefficient2)) <= fairness_bound);
		//An alternative way:
		//fairness_coefficient.push_back(-1);
		//scope.add(bound_fairness);
		//s.add( Sum(scope, fairness_coefficient, 0, 0) );
		//s.add( Sum(scope, fairness_coefficient) == bound_fairness) ;
}


	void run(int verbosity){
		s.parameters.verbosity = verbosity;
		//s.parameters.time_limit = 0.2; // TODO add arg
		//std::cout <<  s << std::endl;
		//s.rewrite();
		s.consolidate();
		if(L > U){
			printf("Lower and upper bound meet. : L = %d ; U = %d\n", L, U);
			result = UNSAT;
		} else {
			result =  s.solve();
		}
	}

	void print_statistics(){
		s.statistics.print_full(std::cout);
	}


	bool isFeasible(){
        if (result )
		{
            return true;
        } else {
            return false;
        }
    }


	void print_and_verify_solution(){

		if (result == SAT)
		{
			std::cout <<  " \n \n c Solution Found! "  << 	std::endl;

			//for ( unsigned int i= 0 ; i< scope.size ; ++i)
			//	std::cout <<  " c Solution value of scope[" << i <<  "]  is " << scope[i].get_solution_int_value() << std::endl;


			int sp_plus= scope[0].get_solution_int_value();
			int sp_minus= scope[1].get_solution_int_value();
			int su_plus= scope[2].get_solution_int_value();
			int su_minus= scope[3].get_solution_int_value();

			//int fairness_sum= scope[4].get_solution_int_value();

			std::cout <<  " c nb protected positive that are predicted positive: " <<  sp_plus << std::endl;
			std::cout <<  " c nb protected negative that are predicted positive: " <<  sp_minus << std::endl;
			std::cout <<  " c nb unprotected positive that are predicted positive: " <<  su_plus << std::endl;
			std::cout <<  " c nb unprotected negative that are predicted positive:  " <<  su_minus << std::endl;
			//std::cout <<  " c nb fairness_sum  " <<  fairness_sum << std::endl;
			//printf("nb_protected_negative : %d\n", nb_protected_negative);
			//printf("nb_unprotected_negative : %d\n", nb_unprotected_negative);
			int accuracy = sp_plus + su_plus -sp_minus - su_minus + nb_protected_negative + nb_unprotected_negative;
			float deltaTPR = ((float)((sp_plus) /(float) (nb_protected - nb_protected_negative)  )  - 	 ((float)(su_plus)/(float) (nb_unprotected - nb_unprotected_negative)  ));
			float deltaFPR = ((float)((sp_minus) /(float) (nb_protected_negative)  )  - 	 ((float)(su_minus)/(float) (nb_unprotected_negative)  ));
			float fairness = fabs(deltaTPR) + fabs(deltaFPR);

			std::cout <<  " c Accuracy ( Number of examamples well classified) is "  << accuracy  << " out of " << nb_protected+ nb_unprotected << 	std::endl;

			std::cout <<  " c EO Unfairness (as float) is "  << fairness << std::endl;
			std::cout <<  " c EO Unfairness tolerance (as float) is "  << fairness_tolerence << std::endl;
			assert (accuracy >= L   ) ;
			assert (accuracy <= U   ) ;
			assert (fabs(fairness - fairness_tolerence) <=  10E-6   ) ;
			//int discrete_fairness= (nb_unprotected * (sp_plus+sp_minus)) - (nb_protected * (su_plus +su_minus) )  ;
			//std::cout <<  " c Fairness (discrete constraint) is "  << discrete_fairness << std::endl;
			std::cout <<  " c Solution Verified"  << 	std::endl;

		}
		else if (result == UNSAT)
			std::cout <<  " c No Solution! "  << 	std::endl;
		else if (result == LIMITOUT)
			std::cout <<  " c Timeout reached! "  << 	std::endl;
		else
			std::cout <<  " c Other status! "  << 	std::endl;
	}


};