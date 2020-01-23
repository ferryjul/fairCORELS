echo '<<<<<------------------------------------ Adult Income [Without sensitive attribute (gender)] --------------------------------->>>>>\n' >> reports/adult_without.txt

python report.py --exp=results --id=1 --attr=1 --m=1  >> reports/adult_without.txt

python report.py --exp=results --id=1 --attr=1 --m=2  >> reports/adult_without.txt

python report.py --exp=results --id=1 --attr=1 --m=3  >> reports/adult_without.txt

python report.py --exp=results --id=1 --attr=1 --m=4  >> reports/adult_without.txt

python report.py --exp=results --id=1 --attr=1 --m=5  >> reports/adult_without.txt

python report.py --exp=results --id=1 --attr=1 --m=6  >> reports/adult_without.txt


echo '<<<<<------------------------------------ Adult Income [With sensitive attribute (gender)] --------------------------------->>>>>\n' >> reports/adult_with.txt

python report.py --exp=results --id=1 --attr=2 --m=1  >> reports/adult_with.txt

python report.py --exp=results --id=1 --attr=2 --m=2  >> reports/adult_with.txt

python report.py --exp=results --id=1 --attr=2 --m=3  >> reports/adult_with.txt

python report.py --exp=results --id=1 --attr=2 --m=4  >> reports/adult_with.txt

python report.py --exp=results --id=1 --attr=2 --m=5  >> reports/adult_with.txt

python report.py --exp=results --id=1 --attr=2 --m=6  >> reports/adult_with.txt



echo '<<<<<------------------------------------ Adult Income [Without sensitive attribute (gender) and relationship] --------------------------------->>>>>\n' >> reports/adult_relationship.txt

python report.py --exp=results --id=6 --attr=1 --m=1  >> reports/adult_relationship.txt

python report.py --exp=results --id=6 --attr=1 --m=2  >> reports/adult_relationship.txt

python report.py --exp=results --id=6 --attr=1 --m=3  >> reports/adult_relationship.txt

python report.py --exp=results --id=6 --attr=1 --m=4  >> reports/adult_relationship.txt

python report.py --exp=results --id=6 --attr=1 --m=5  >> reports/adult_relationship.txt

python report.py --exp=results --id=6 --attr=1 --m=6  >> reports/adult_relationship.txt



echo '<<<<<------------------------------------ Adult Income [Without sensitive attribute (marital status)] --------------------------------->>>>>\n' >> reports/adult_marital.txt

python report.py --exp=results --id=5 --attr=1 --m=1  >> reports/adult_marital.txt

python report.py --exp=results --id=5 --attr=1 --m=2  >> reports/adult_marital.txt

python report.py --exp=results --id=5 --attr=1 --m=3  >> reports/adult_marital.txt

python report.py --exp=results --id=5 --attr=1 --m=4  >> reports/adult_marital.txt

python report.py --exp=results --id=5 --attr=1 --m=5  >> reports/adult_marital.txt

python report.py --exp=results --id=5 --attr=1 --m=6  >> reports/adult_marital.txt

