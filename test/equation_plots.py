"""
Simple script to check variations are coherent
"""

import numpy as np
from matplotlib.pyplot import *

def hex(X, p):
    return X/(X+p)

def differenciation(X, affinity, differenciation_rate):
    return affinity*np.exp(-differenciation_rate * X)


p1 = 1

x = np.arange(0, 10, 1)
print(x)


plot(x, [differenciation(k, 1, 0.1) for k in x])