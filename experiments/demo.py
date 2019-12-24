import numpy as np 


epsilons = np.arange(0.91, 1.001, 0.001)

epsilons = [0.0] + list(epsilons)

epsilons = [round(x,3) for x in epsilons]

print(epsilons)

print(len(epsilons))