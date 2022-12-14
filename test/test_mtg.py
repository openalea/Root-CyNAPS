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
           p: str = 'influx_N'
           ):
    props = g.property(p)
    max_scale = g.max_scale()
    plot_range = [props[vid] for vid in g.vertices(scale=max_scale) if props[vid] != 0]

    scene = plot_mtg(g,
                     prop_cmap=p,
                     lognorm=False,  # to avoid issues with negative values
                     vmin=min(plot_range),
                     vmax=max(plot_range)
                     )
    pgl.Viewer.display(scene)


def print_g(g,
            select
            ):
    # extract MTG properties only once
    props = g.properties()
    extract = [props[k] for k in select]

    max_scale = g.max_scale()
    for vid in g.vertices(scale=max_scale):
        # print for each segment selected properties in select
        print(vid, end=' ')
        for k in range(len(extract)):
            print(select[k] + ' : ', end=' ')
            print(f"{extract[k][vid]:4.10f}", end=' ')
        print('')


def test_nitrogen(n=10):
    g = test_mtg()

    # Initialization of state variable
    g = init_N(g)

    for i in range(n):
        g = transport_N(g)
        g = update_N(g)

    plot_N(g, p='C_hexose_root')
    select = ['influx_N', 'z1', 'C_hexose_root']
    print_g(g, select)

    return g


# Test execution
# TODO : use timeit to compare calculation times between mtg access sequences

test_nitrogen()
input('end? ')
