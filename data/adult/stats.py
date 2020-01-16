import pandas as pd
import numpy as np


"""df = pd.read_csv('./adult_rules_full.csv')

n_maj = np.sum(df['gender_Male'] == 1)
maj_pos = np.sum(np.logical_and(df['gender_Male'] == 1, df.income == 1))

n_min = np.sum(df['gender_Female'] == 1)
min_pos = np.sum(np.logical_and(df['gender_Female'] == 1, df.income == 1))



print('maj rate  {}'.format(float(maj_pos)/n_maj))
print('min rate  {}'.format(float(min_pos)/n_min))"""


df = pd.read_csv('./adult_clean.csv')

n_maj = np.sum(df['gender'] == 'Male')
maj_pos = np.sum(np.logical_and(df['gender'] == 'Male', df.income == 1))
maj_neg = np.sum(np.logical_and(df['gender'] == 'Male', df.income == 0))

n_min = np.sum(df['gender'] == 'Female')
min_pos = np.sum(np.logical_and(df['gender'] == 'Female', df.income == 1))


print("========")
print('maj pos rate  {}'.format(float(maj_pos)/n_maj))
#print('maj neg rate  {}'.format(float(maj_neg)/n_maj))

print('min pos rate  {}'.format(float(min_pos)/n_min))

