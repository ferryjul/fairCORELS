echo "OPT PER EPSILON"
echo "0/4"
python3 gen_results_OPTPEREPS.py --dataset=german_credit --maxTime=2400 --metric=1 --save=True --show=0 --reverseEps=1
echo "1/4"
python3 gen_results_OPTPEREPS.py --dataset=german_credit --maxTime=2400 --metric=3 --save=True --show=0 --reverseEps=1
echo "2/4"
python3 gen_results_OPTPEREPS.py --dataset=german_credit --maxTime=2400 --metric=4 --save=True --show=0 --reverseEps=1
echo "3/4"
python3 gen_results_OPTPEREPS.py --dataset=german_credit --maxTime=2400 --metric=5 --save=True --show=0 --reverseEps=1
echo "4/4"

echo "CACHE PER EPSILON"
echo "0/4"
python3 gen_results_CACHEPEREPS.py --dataset=german_credit --maxTime=2400 --metric=1 --save=True --show=0 --displayNbInstances=0 --reverseEps=1
echo "1/4"
python3 gen_results_CACHEPEREPS.py --dataset=german_credit --maxTime=2400 --metric=3 --save=True --show=0 --displayNbInstances=0 --reverseEps=1
echo "2/4"
python3 gen_results_CACHEPEREPS.py --dataset=german_credit --maxTime=2400 --metric=4 --save=True --show=0 --displayNbInstances=0 --reverseEps=1
echo "3/4"
python3 gen_results_CACHEPEREPS.py --dataset=german_credit --maxTime=2400 --metric=5 --save=True --show=0 --displayNbInstances=0 --reverseEps=1
echo "4/4"

echo "OPT PER EPSILON"
echo "0/4"
python3 gen_results_OPTPEREPS.py --dataset=compas --maxTime=1200 --metric=1 --save=True --show=0 --reverseEps=1
echo "1/4"
python3 gen_results_OPTPEREPS.py --dataset=compas --maxTime=1200 --metric=3 --save=True --show=0 --reverseEps=1
echo "2/4"
python3 gen_results_OPTPEREPS.py --dataset=compas --maxTime=1200 --metric=4 --save=True --show=0 --reverseEps=1
echo "3/4"
python3 gen_results_OPTPEREPS.py --dataset=compas --maxTime=1200 --metric=5 --save=True --show=0 --reverseEps=1
echo "4/4"

echo "CACHE PER EPSILON"
echo "0/4"
python3 gen_results_CACHEPEREPS.py --dataset=compas --maxTime=1200 --metric=1 --save=True --show=0 --displayNbInstances=0 --reverseEps=1
echo "1/4"
python3 gen_results_CACHEPEREPS.py --dataset=compas --maxTime=1200 --metric=3 --save=True --show=0 --displayNbInstances=0 --reverseEps=1
echo "2/4"
python3 gen_results_CACHEPEREPS.py --dataset=compas --maxTime=1200 --metric=4 --save=True --show=0 --displayNbInstances=0 --reverseEps=1
echo "3/4"
python3 gen_results_CACHEPEREPS.py --dataset=compas --maxTime=1200 --metric=5 --save=True --show=0 --displayNbInstances=0 --reverseEps=1
echo "4/4"