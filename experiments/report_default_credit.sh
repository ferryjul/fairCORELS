echo '<<<<<------------------------------------ Default Credit [Without sensitive attribute (gender)] --------------------------------->>>>>\n' >> reports/default_credit_without.txt

python report.py --exp=results --id=4 --attr=1 --m=1  >> reports/default_credit_without.txt

python report.py --exp=results --id=4 --attr=1 --m=2  >> reports/default_credit_without.txt

python report.py --exp=results --id=4 --attr=1 --m=3  >> reports/default_credit_without.txt

python report.py --exp=results --id=4 --attr=1 --m=4  >> reports/default_credit_without.txt

python report.py --exp=results --id=4 --attr=1 --m=5  >> reports/default_credit_without.txt

python report.py --exp=results --id=4 --attr=1 --m=6  >> reports/default_credit_without.txt


echo '<<<<<------------------------------------ Default Credit [With sensitive attribute (gender)] --------------------------------->>>>>\n' >> reports/default_credit_with.txt

python report.py --exp=results --id=4 --attr=2 --m=1  >> reports/default_credit_with.txt

python report.py --exp=results --id=4 --attr=2 --m=2  >> reports/default_credit_with.txt

python report.py --exp=results --id=4 --attr=2 --m=3  >> reports/default_credit_with.txt

python report.py --exp=results --id=4 --attr=2 --m=4  >> reports/default_credit_with.txt

python report.py --exp=results --id=4 --attr=2 --m=5  >> reports/default_credit_with.txt

python report.py --exp=results --id=4 --attr=2 --m=6  >> reports/default_credit_with.txt

