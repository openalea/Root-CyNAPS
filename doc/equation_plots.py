"""
Simple script to check variations are coherent
"""

import numpy as np
from matplotlib.pyplot import *

def MM(X, p, S):
    return X*S/(X+p)

def differenciation(X, affinity, differenciation_rate):
    return affinity*np.exp(-differenciation_rate * X)


p1 = 1

x = np.arange(0, 1, 0.01)

plot(x, [MM(k, 1, 0.1) for k in x])