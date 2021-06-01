echo "LEARNING QUALITY TABLE GERMAN CREDIT "
echo "0/4"
python3 gen_results_ACC_GENERALIZATION.py --mode=summary_table --dataset=german_credit --metric=1
echo "1/4"
python3 gen_results_ACC_GENERALIZATION.py --mode=summary_table --dataset=german_credit --metric=3
echo "2/4"
python3 gen_results_ACC_GENERALIZATION.py --mode=summary_table --dataset=german_credit --metric=4
echo "3/4"
python3 gen_results_ACC_GENERALIZATION.py --mode=summary_table --dataset=german_credit --metric=5
echo "4/4"

echo "LEARNING QUALITY TABLE COMPAS"
echo "0/4"
python3 gen_results_ACC_GENERALIZATION.py --mode=summary_table --dataset=compas --metric=1
echo "1/4"
python3 gen_results_ACC_GENERALIZATION.py --mode=summary_table --dataset=compas --metric=3
echo "2/4"
python3 gen_results_ACC_GENERALIZATION.py --mode=summary_table --dataset=compas --metric=4
echo "3/4"
python3 gen_results_ACC_GENERALIZATION.py --mode=summary_table --dataset=compas --metric=5
echo "4/4"

echo "MERGING"
python3 merge_learning_tables.py --dataset=compas
python3 merge_learning_tables.py --dataset=german_credit
echo "DONE"