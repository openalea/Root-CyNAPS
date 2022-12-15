"""

"""

import pickle
import numpy as np
from openalea.mtg import *
from rhizodep.nitrogen import continuous_vessels
from output_display import plot_N, print_g



def test_mtg():
    with open('inputs/root00119.pckl', 'rb') as f:
        g = pickle.load(f)
    return g




def test_nitrogen(n=10):
    g = test_mtg()

    # Initialization of state variable
    rs = continuous_vessels(g)

    for i in range(n):
        rs.transport_N()
        rs.update_N()
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
