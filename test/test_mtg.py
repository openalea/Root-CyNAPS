"""

"""

import pickle
from openalea.mtg import *


def test_mtg():
    with open('test/inputs/root00478.pckl', 'rb') as f:
        g = pickle.load(f)
    return g

if __name__ == '__main__':
    print(test_mtg().properties())
    print([k for k in test_mtg().properties()])