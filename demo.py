import numpy as np 

epsilon_low_regime = np.linspace(0.89, 0.949, num=10) 
epsilon_high_regime = np.linspace(0.95, 0.999, num=60)
epsilon_range = [0.0] + [x for x in epsilon_low_regime] + [x for x in epsilon_high_regime] + [1.0]

print(len(epsilon_range))