echo '<<<<<------------------------------------ Compas [Without sensitive attribute (race)] --------------------------------->>>>>\n' >> reports/compas_without.txt

python report.py --exp=results --id=2 --attr=1 --m=1  >> reports/compas_without.txt

python report.py --exp=results --id=2 --attr=1 --m=2  >> reports/compas_without.txt

python report.py --exp=results --id=2 --attr=1 --m=3  >> reports/compas_without.txt

python report.py --exp=results --id=2 --attr=1 --m=4  >> reports/compas_without.txt

python report.py --exp=results --id=2 --attr=1 --m=5  >> reports/compas_without.txt

python report.py --exp=results --id=2 --attr=1 --m=6  >> reports/compas_without.txt


echo '<<<<<------------------------------------ Compas [With sensitive attribute (race)] --------------------------------->>>>>\n' >> reports/compas_with.txt

python report.py --exp=results --id=2 --attr=2 --m=1  >> reports/compas_with.txt

python report.py --exp=results --id=2 --attr=2 --m=2  >> reports/compas_with.txt

python report.py --exp=results --id=2 --attr=2 --m=3  >> reports/compas_with.txt

python report.py --exp=results --id=2 --attr=2 --m=4  >> reports/compas_with.txt

python report.py --exp=results --id=2 --attr=2 --m=5  >> reports/compas_with.txt

python report.py --exp=results --id=2 --attr=2 --m=6  >> reports/compas_with.txt

