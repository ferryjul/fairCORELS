echo '<<<<<------------------------------------ German Credit [Without sensitive attribute (age)] --------------------------------->>>>>\n' >> reports/german_credit_without.txt

python report.py --exp=results --id=3 --attr=1 --m=1  >> reports/german_credit_without.txt

python report.py --exp=results --id=3 --attr=1 --m=2  >> reports/german_credit_without.txt

python report.py --exp=results --id=3 --attr=1 --m=3  >> reports/german_credit_without.txt

python report.py --exp=results --id=3 --attr=1 --m=4  >> reports/german_credit_without.txt

python report.py --exp=results --id=3 --attr=1 --m=5  >> reports/german_credit_without.txt

python report.py --exp=results --id=3 --attr=1 --m=6  >> reports/german_credit_without.txt


echo '<<<<<------------------------------------ German Credit [With sensitive attribute (age)] --------------------------------->>>>>\n' >> reports/german_credit_with.txt

python report.py --exp=results --id=3 --attr=2 --m=1  >> reports/german_credit_with.txt

python report.py --exp=results --id=3 --attr=2 --m=2  >> reports/german_credit_with.txt

python report.py --exp=results --id=3 --attr=2 --m=3  >> reports/german_credit_with.txt

python report.py --exp=results --id=3 --attr=2 --m=4  >> reports/german_credit_with.txt

python report.py --exp=results --id=3 --attr=2 --m=5  >> reports/german_credit_with.txt

python report.py --exp=results --id=3 --attr=2 --m=6  >> reports/german_credit_with.txt

