from io import StringIO
import numpy as np
import pandas as pd
import math

from collections import Counter
from imblearn.under_sampling import NearMiss


import requests
from requests import RequestException
import logging

logger = logging.getLogger(__name__)

ADULT_URLS = [ 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
]

def fetch_adult():
    
    url_data = ADULT_URLS[0]
    url_test = ADULT_URLS[1]

    raw_features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'maritalStatus',
                    'occupation', 'relationship', 'race', 'gender', 'capitalGain', 'capitalLoss',
                    'hoursPerWeek', 'nativeCountry', 'income']
    
    try:
        resp_data = requests.get(url_data)
        resp_data.raise_for_status()
        resp_test = requests.get(url_test)
        resp_test.raise_for_status()
    except RequestException:
        logger.exception("Impossible to download the file, URL may be out of service")
        raise
    

    df_data = pd.read_csv(StringIO(resp_data.text), names=raw_features, delimiter=', ', engine='python').replace({'?': np.nan}).dropna()
    df_test = pd.read_csv(StringIO(resp_test.text), names=raw_features, delimiter=', ', engine='python').replace({'?': np.nan}).dropna()


    dataset = pd.concat([df_data, df_test])
    
    def basic_clean():
        dropList = ["fnlwgt", "education_num"]
        dataset['income']=dataset['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
        dataset.drop(labels=dropList, axis = 1, inplace = True)    

    def process_education():
        education = []
        for index, row in dataset.iterrows():

            if(row['education'] in ["10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th", "Preschool"] ):
                education.append("dropout")

            if(row['education'] in ["Assoc-acdm", "Assoc-voc"] ):
                education.append("associates")

            if(row['education'] in ["Bachelors"] ):
                education.append("bachelors")

            if(row['education'] in ["Masters", "Doctorate"] ):
                education.append("masters_doctorate")

            if(row['education'] in ["HS-grad", "Some-college"] ):
                education.append("hs_grad")

            if(row['education'] in ["Prof-school"] ):
                education.append("prof_school")
            
        dataset['education'] = education

    def process_workclass():
        workclass = []
        for index, row in dataset.iterrows():

            if(row['workclass'] in ["Federal-gov"] ):
                workclass.append("fedGov")

            if(row['workclass'] in ["Local-gov", "State-gov"] ):
                workclass.append("otherGov")

            if(row['workclass'] in ["Private"] ):
                workclass.append("private")

            if(row['workclass'] in ["Self-emp-inc", "Self-emp-not-inc"] ):
                workclass.append("selfEmployed")

            if(row['workclass'] in ["Without-pay", "Never-worked" ] ):
                workclass.append("unEmployed")

        dataset['workclass'] = workclass

    def process_occupation():
        occupation = []
        for index, row in dataset.iterrows():

            if(row['occupation'] in ["Craft-repair", "Farming-fishing","Handlers-cleaners", "Machine-op-inspct", "Transport-moving"] ):
                occupation.append("blueCollar")

            if(row['occupation'] in ["Exec-managerial"] ):
                occupation.append("whiteCollar")
            
            if(row['occupation'] in ["Sales"] ):
                occupation.append("sales")

            if(row['occupation'] in ["Prof-specialty"] ):
                occupation.append("professional")

            if(row['occupation'] in ["Tech-support", "Protective-serv", "Armed-Forces", "Other-service", "Priv-house-serv", "Adm-clerical"] ):
                    occupation.append("other")

        dataset['occupation'] = occupation

    def process_marital():
        marital = []
        for index, row in dataset.iterrows():

            if(row['maritalStatus'] in ["Never-married"] ):
                marital.append("single")

            if(row['maritalStatus'] in ["Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse" ] ):
                marital.append("married")

        dataset['maritalStatus'] = marital

    def process_relationship():
        relationship = []
        for index, row in dataset.iterrows():

            if(row['relationship'] in ["Husband"] ):
                relationship.append("husband")

            if(row['relationship'] in ["Wife"] ):
                relationship.append("wife")

            if(row['relationship'] in ["Not-in-family"] ):
                relationship.append("notInFamily")

            if(row['relationship'] in ["Own-child"] ):
                relationship.append("ownChild")

            if(row['relationship'] in ["Unmarried"] ):
                relationship.append("unmarried")

            if(row['relationship'] in ["Other-relative"] ):
                relationship.append("otherRelative")
            
        dataset['relationship'] = relationship

    def process_country():
        native_country = []
        for index, row in dataset.iterrows():

            if(row['nativeCountry'] in ["United-States"] ):
                native_country.append("USA")
            else: 
                native_country.append("notUSA")

        dataset['nativeCountry'] = native_country

    def process_race():
        race = []
        for index, row in dataset.iterrows():

            if(row['race'] in ["White"] ):
                race.append("White")

            if(row['race'] in ["Black"] ):
                race.append("Black")

            if(row['race'] in ["Asian-Pac-Islander"] ):
                race.append("AsianPacIslander")

            if(row['race'] in ["Amer-Indian-Eskimo"] ):
                race.append("AmerIndianEskimo")
            
            if(row['race'] in ["Other"] ):
                race.append("Other")
            
        dataset['race'] = race

    
    def process_numerical():

        num_cols = ['hoursPerWeek', 'age', 'capitalGain', 'capitalLoss']
        for num in num_cols:
            dataset[num] = dataset[num].astype(float)

        cols_to_scale_none = ['hoursPerWeek', 'age']
        
        cols_other = ['capitalGain', 'capitalLoss']
        
        
        def four_cut_off(x, low, high):
            out = ""
            if (x <= low ):
                out = "low"

            if ((x > low) and (x < high)):
                out = "middle"
            
            if (x >= high ):
                out = "high"

            return out
        
        
        for num in cols_to_scale_none:
            dataset[num] = dataset[num].apply(four_cut_off, low=np.percentile(dataset[num], 25),   high=np.percentile(dataset[num], 75) )

        for num in cols_other:
            dataset[num] = dataset[num].apply(lambda x : "high" if x > 0 else "low")

        

    basic_clean()
    process_education()
    process_workclass()
    process_occupation()
    process_marital()
    process_relationship()
    process_country()
    process_race()
    process_numerical()

    y = dataset.income.values
    dataset.drop(labels=["income"], axis=1, inplace=True)
    X = pd.get_dummies(dataset)
    columns = list(X)

    dataset = pd.DataFrame(X, columns=columns)
    dataset['income'] = y
    dataset.to_csv("./data/adult_full.csv", encoding='utf-8', index=False)

    print('None -----', sorted(Counter(y).items()))

    nm3 = NearMiss(version=2, random_state=42, n_jobs=-1)
    X_resampled, y_resampled = nm3.fit_resample(X, y)
    print('NearMiss ------', sorted(Counter(y_resampled).items()))
   

    dataset = pd.DataFrame(X_resampled, columns=columns)
    dataset['income'] = y_resampled
    dataset.to_csv("./data/adult_undersampled.csv", encoding='utf-8', index=False)


fetch_adult()