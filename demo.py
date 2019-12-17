import numpy as np 
import pandas as pd
import time

from joblib import Parallel, delayed, parallel_backend


"""epsilon_low_regime = np.linspace(0.89, 0.949, num=10) 
epsilon_high_regime = np.linspace(0.95, 0.999, num=60)
epsilon_range = [0.0] + [x for x in epsilon_low_regime] + [x for x in epsilon_high_regime] + [1.0]

print(len(epsilon_range))"""

a = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3],[1, 2, 3]]

def doub(x):
    return 2*x


def unit_loop(y):
    print('--------->')
    res = Parallel(n_jobs=3)(delayed(doub)(x=i) for i in y)

    row = {
        "first" : res[0],
        "second" : res[1],
        "third" : res[2],
    }
    time.sleep(1)
    return row

def main_loop():
    filename = './demo.csv'
    row_list = Parallel(n_jobs=4)(delayed(unit_loop)(x) for x in a)
    df = pd.DataFrame(row_list)
    df.to_csv(filename, encoding='utf-8', index=False)


main_loop()