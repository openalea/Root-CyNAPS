"""

"""

import pickle
import numpy as np
from openalea.mtg import *
from output_display import plot_N, print_g



def test_mtg():
    with open('test/inputs/root00239.pckl', 'rb') as f:
        g = pickle.load(f)
    return g

if __name__ == '__main__':
    print(test_mtg().properties())
    print([k for k in test_mtg().properties()])