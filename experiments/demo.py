import numpy as np 


epsilon_range = np.arange(0.95, 1.001, 0.001)

base = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

epsilon_range = base + list(epsilon_range)

epsilon_range = [round(x,3) for x in epsilon_range]

print(epsilon_range)

print(len(epsilon_range))