echo "OPT PER EPSILON"
echo "0/4"
python3 gen_results_OPTPEREPS.py --dataset=compas --maxTime=1200 --metric=1 --save=True --show=0 
echo "1/4"
python3 gen_results_OPTPEREPS.py --dataset=compas --maxTime=1200 --metric=3 --save=True --show=0 
echo "2/4"
python3 gen_results_OPTPEREPS.py --dataset=compas --maxTime=1200 --metric=4 --save=True --show=0 
echo "3/4"
python3 gen_results_OPTPEREPS.py --dataset=compas --maxTime=1200 --metric=5 --save=True --show=0 
echo "4/4"

echo "OPT PER TIME"
echo "0/4"
python3 gen_results_OPTPERTIME.py --dataset=compas --epsilons 0.98 0.985 0.99 0.995 --save=True --show=0 --metric=1 
echo "1/4"
python3 gen_results_OPTPERTIME.py --dataset=compas --epsilons 0.98 0.985 0.99 0.995 --save=True --show=0 --metric=3 
echo "2/4" 
python3 gen_results_OPTPERTIME.py --dataset=compas --epsilons 0.98 0.985 0.99 0.995 --save=True --show=0 --metric=4 
echo "3/4"
python3 gen_results_OPTPERTIME.py --dataset=compas --epsilons 0.98 0.985 0.99 0.995 --save=True --show=0 --metric=5 
echo "4/4"

echo "SCORE PER TIME"
echo "0/4"
python3 gen_results_SCOREPERTIME.py --dataset=compas --epsilons 0.98 0.985 0.99 0.995 --save=True --show=0 --metric=1
echo "1/4"
python3 gen_results_SCOREPERTIME.py --dataset=compas --epsilons 0.98 0.985 0.99 0.995 --save=True --show=0 --metric=3
echo "2/4" 
python3 gen_results_SCOREPERTIME.py --dataset=compas --epsilons 0.98 0.985 0.99 0.995 --save=True --show=0 --metric=4
echo "3/4"
python3 gen_results_SCOREPERTIME.py --dataset=compas --epsilons 0.98 0.985 0.99 0.995 --save=True --show=0 --metric=5
echo "4/4"

echo "CACHE PER EPSILON"
echo "0/4"
python3 gen_results_CACHEPEREPS.py --dataset=compas --maxTime=1200 --metric=1 --save=True --show=0 --displayNbInstances=0
echo "1/4"
python3 gen_results_CACHEPEREPS.py --dataset=compas --maxTime=1200 --metric=3 --save=True --show=0 --displayNbInstances=0
echo "2/4"
python3 gen_results_CACHEPEREPS.py --dataset=compas --maxTime=1200 --metric=4 --save=True --show=0 --displayNbInstances=0
echo "3/4"
python3 gen_results_CACHEPEREPS.py --dataset=compas --maxTime=1200 --metric=5 --save=True --show=0 --displayNbInstances=0
echo "4/4"

echo "FILTERING TABLE"
echo "0/4"
python3 gen_results_TABLE.py --dataset=compas --maxTime=1200 --metric=1
echo "1/4"
python3 gen_results_TABLE.py --dataset=compas --maxTime=1200 --metric=3
echo "2/4"
python3 gen_results_TABLE.py --dataset=compas --maxTime=1200 --metric=4
echo "3/4"
python3 gen_results_TABLE.py --dataset=compas --maxTime=1200 --metric=5
echo "4/4"

echo "LEARNING QUALITY TABLE"
echo "0/4"
python3 gen_results_ACC_GENERALIZATION.py --mode=summary_table --dataset=compas --metric=1
echo "1/4"
python3 gen_results_ACC_GENERALIZATION.py --mode=summary_table --dataset=compas --metric=3
echo "2/4"
python3 gen_results_ACC_GENERALIZATION.py --mode=summary_table --dataset=compas --metric=4
echo "3/4"
python3 gen_results_ACC_GENERALIZATION.py --mode=summary_table --dataset=compas --metric=5
echo "4/4"
python3 merge_learning_tables.py --dataset=compas