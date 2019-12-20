echo '<<<<<------------------------------------Adult Income--------------------------------->>>>>\n' >> details.txt
echo '|||||||------------------------------------Adult without sensitive attribute \n' >> details.txt

python report.py --exp=results_1M --id=1 --attr=1 --m=1 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=1 --attr=1 --m=2 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=1 --attr=1 --m=3 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=1 --attr=1 --m=4 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=1 --attr=1 --m=5 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=1 --attr=1 --m=6 >> details.txt
echo '\n' >> details.txt

echo '|||||||------------------------------------Adult with sensitive attribute \n' >> details.txt

python report.py --exp=results_1M --id=1 --attr=2 --m=1 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=1 --attr=2 --m=2 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=1 --attr=2 --m=3 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=1 --attr=2 --m=4 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=1 --attr=2 --m=5 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=1 --attr=2 --m=6 >> details.txt
echo '\n' >> details.txt

echo '<<<<<------------------------------------Compas--------------------------------->>>>>\n' >> details.txt
echo '|||||||------------------------------------Compas without sensitive attribute \n' >> details.txt

python report.py --exp=results_1M --id=2 --attr=1 --m=1 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=2 --attr=1 --m=2 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=2 --attr=1 --m=3 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=2 --attr=1 --m=4 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=2 --attr=1 --m=5 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=2 --attr=1 --m=6 >> details.txt
echo '\n' >> details.txt

echo '|||||||------------------------------------Compas with sensitive attribute \n' >> details.txt

python report.py --exp=results_1M --id=2 --attr=2 --m=1 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=2 --attr=2 --m=2 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=2 --attr=2 --m=3 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=2 --attr=2 --m=4 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=2 --attr=2 --m=5 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=2 --attr=2 --m=6 >> details.txt
echo '\n' >> details.txt


echo '<<<<<------------------------------------Default Credit--------------------------------->>>>>\n' >> details.txt
echo '|||||||------------------------------------Default Credit without sensitive attribute \n' >> details.txt

python report.py --exp=results_1M --id=4 --attr=1 --m=1 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=4 --attr=1 --m=2 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=4 --attr=1 --m=3 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=4 --attr=1 --m=4 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=4 --attr=1 --m=5 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=4 --attr=1 --m=6 >> details.txt
echo '\n' >> details.txt

echo '|||||||------------------------------------Default Credit with sensitive attribute \n' >> details.txt

python report.py --exp=results_1M --id=4 --attr=2 --m=1 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=4 --attr=2 --m=2 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=4 --attr=2 --m=3 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=4 --attr=2 --m=4 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=4 --attr=2 --m=5 >> details.txt
echo '\n' >> details.txt
python report.py --exp=results_1M --id=4 --attr=2 --m=6 >> details.txt
echo '\n' >> details.txt