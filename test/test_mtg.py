from openalea.mtg import *
from openalea.mtg.traversal import post_order
import pickle
import numpy as np
from rhizodep.nitrogen import init_N, transport_N, update_N
from rhizodep.tools import plot_mtg
import openalea.plantgl.all as pgl


def test_mtg():
    with open('inputs/root00119.pckl', 'rb') as f:
        g = pickle.load(f)
    return g


def plot_N(g,
           p : str = 'influx_N'
           ):

    scene = plot_mtg(g,
             prop_cmap=p)
    pgl.Viewer.display(scene)


def test_nitrogen(n=10):
    g = test_mtg()

    # Initiatisation of state variable
    g = init_N(g)

    for i in range(n):
        g = transport_N(g)
        g = update_N(g)

    plot_N(g)
    return g


# Test execution
prop = 'influx_N'
g = test_nitrogen()

while prop != 'n':
    plot_N(g, p=prop)
    print(g.property(prop).values)
    prop = input('next property? : ')
