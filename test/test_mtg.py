"""

"""

import pickle
from openalea.mtg import *


def import_mtg():
    with open('test/inputs/root00478.pckl', 'rb') as f:
        g = pickle.load(f)
    return g

if __name__ == '__main__':
    print(import_mtg().properties())
    print([k for k in import_mtg().properties()])
