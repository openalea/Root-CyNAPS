"""

"""

import pickle
import numpy as np
from openalea.mtg import *
from rhizodep.nitrogen import init_N, transport_N, update_N
from output_display import plot_N, print_g



def test_mtg():
    with open('inputs/root00119.pckl', 'rb') as f:
        g = pickle.load(f)
    return g




def test_nitrogen(n=10):
    g = test_mtg()

    # Initialization of state variable
    g = init_N(g)

    for i in range(n):
        g = transport_N(g)
        g = update_N(g)
        #print_g(g, select, vertice=19)
    plot_N(g, p='influx_Nm')

    print_g(g)
    print(g.node(0).xylem_Nm, g.node(0).xylem_volume)

    return g


# Test execution
# TODO : use timeit to compare calculation times between mtg access sequences

if __name__ == '__main__':
    test_nitrogen()
    input('end? ')
