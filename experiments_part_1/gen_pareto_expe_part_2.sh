# faircorels | all metric | adult_with
python compute_pareto.py --method=1 --dataset=5 --metric=1
python compute_pareto.py --method=1 --dataset=5 --metric=2
python compute_pareto.py --method=1 --dataset=5 --metric=3
python compute_pareto.py --method=1 --dataset=5 --metric=4
python compute_pareto.py --method=1 --dataset=5 --metric=5
python compute_pareto.py --method=1 --dataset=5 --metric=6


# faircorels | all metric | compas_with
python compute_pareto.py --method=1 --dataset=4 --metric=1
python compute_pareto.py --method=1 --dataset=4 --metric=2
python compute_pareto.py --method=1 --dataset=4 --metric=3
python compute_pareto.py --method=1 --dataset=4 --metric=4
python compute_pareto.py --method=1 --dataset=4 --metric=5
python compute_pareto.py --method=1 --dataset=4 --metric=6

#-----------------------------------------------------------------------

# laftr | metric 1, 4, 5 | adult_with
python compute_pareto.py --method=2 --dataset=5 --metric=1
python compute_pareto.py --method=2 --dataset=5 --metric=4
python compute_pareto.py --method=2 --dataset=5 --metric=5

# laftr | metric 1, 4, 5 | compas_with
python compute_pareto.py --method=2 --dataset=4 --metric=1
python compute_pareto.py --method=2 --dataset=4 --metric=4
python compute_pareto.py --method=2 --dataset=4 --metric=5


#-----------------------------------------------------------------------

# zafar | metric 1, 4, 5 | adult_with
python compute_pareto.py --method=3 --dataset=5 --metric=1
python compute_pareto.py --method=3 --dataset=5 --metric=4
python compute_pareto.py --method=3 --dataset=5 --metric=5

# zafar | metric 1, 4, 5 | compas_with
python compute_pareto.py --method=3 --dataset=4 --metric=1
python compute_pareto.py --method=3 --dataset=4 --metric=4
python compute_pareto.py --method=3 --dataset=4 --metric=5

