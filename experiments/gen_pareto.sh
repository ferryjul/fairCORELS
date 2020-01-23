# faircorels | all metric | adult_no_relationship
python compute_pareto.py --method=1 --dataset=1 --metric=1
python compute_pareto.py --method=1 --dataset=1 --metric=2
python compute_pareto.py --method=1 --dataset=1 --metric=3
python compute_pareto.py --method=1 --dataset=1 --metric=4
python compute_pareto.py --method=1 --dataset=1 --metric=5
python compute_pareto.py --method=1 --dataset=1 --metric=6


# faircorels | all metric | compas_without
python compute_pareto.py --method=1 --dataset=2 --metric=1
python compute_pareto.py --method=1 --dataset=2 --metric=2
python compute_pareto.py --method=1 --dataset=2 --metric=3
python compute_pareto.py --method=1 --dataset=2 --metric=4
python compute_pareto.py --method=1 --dataset=2 --metric=5
python compute_pareto.py --method=1 --dataset=2 --metric=6

#-----------------------------------------------------------------------

# laftr | metric 1, 4, 5 | adult_no_relationship
python compute_pareto.py --method=2 --dataset=1 --metric=1
python compute_pareto.py --method=2 --dataset=1 --metric=4
python compute_pareto.py --method=2 --dataset=1 --metric=5

# laftr | metric 1, 4, 5 | compas_without
python compute_pareto.py --method=2 --dataset=2 --metric=1
python compute_pareto.py --method=2 --dataset=2 --metric=4
python compute_pareto.py --method=2 --dataset=2 --metric=5


#-----------------------------------------------------------------------

# zafar | metric 1, 4, 5 | adult_no_relationship
python compute_pareto.py --method=3 --dataset=1 --metric=1
python compute_pareto.py --method=3 --dataset=1 --metric=4
python compute_pareto.py --method=3 --dataset=1 --metric=5

# zafar | metric 1, 4, 5 | compas_without
python compute_pareto.py --method=3 --dataset=2 --metric=1
python compute_pareto.py --method=3 --dataset=2 --metric=4
python compute_pareto.py --method=3 --dataset=2 --metric=5

