import pandas as pd
import numpy as np


df = pd.read_csv('./compas_rules_full.csv')

maj_pos = np.mean(np.logical_and(df['race_Caucasian'] == 1, df.two_year_recid == 0))
min_pos = np.mean(np.logical_and(df['race_African-American'] == 1, df.two_year_recid == 0))



print('maj rate  {}'.format(maj_pos))
print('min rate  {}'.format(min_pos))

