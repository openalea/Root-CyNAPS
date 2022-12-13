from openalea.mtg import *
from openalea.mtg.traversal import post_order
import pickle
import numpy as np


def init_N(g,
           soil_N: float = 0.1,
           N: float = 0.1,
           xylem_N: float = 0.1,
           influx_N: float = 0,
           loading_N: float = 0):
    """

    :param g:
    :param kwds:
    :return:
    """
    # Variable initialisation
    # We define "root" as the starting point of the loop below
    keywords = dict(soil_N=soil_N,
                    N=N,
                    xylem_N=xylem_N,
                    influx_N=influx_N,
                    loading_N=loading_N)
    props = g.properties()
    for name in keywords:
        props.setdefault(name, {})

    max_scale = g.max_scale()
    for vid in g.vertices(scale=max_scale):
        for name, value in keywords.items():
            props[name][vid] = value

    return g

# Example of long calculation time related to repeating calls
'''
def init_N2(g,
           soil_N : float = 0.1,
           N : float = 0.1,
           xylem_N : float = 0.1,
           influx_N : float = 0,
           loading_N : float = 0):
    """

    :param g:
    :param kwds:
    :return:
    """
    # Variable initialisation
    # We define "root" as the starting point of the loop below
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    # We travel in the MTG from the root tips to the base:
    for vid in post_order(g, root):
        # We define the current root element as n:
        n = g.node(vid)
        n.soil_N = soil_N
        n.N = N
        n.influx_N = influx_N
        n.xylem_N = xylem_N
        n.loading_N = loading_N
    return g
'''


def transport_N(g,
                affinity_N_root : float = 0.01,
                vmax_N_root : float = 0.5,
                affinity_N_xylem : float = 10,
                vmax_N_xylem : float = 0.1,
                xylem_to_root : float = 0.2,
                time_step : int = 3600
                ):
    # We define "root" as the starting point of the loop below:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    # We travel in the MTG from the root tips to the base:
    for vid in post_order(g, root):
        # We define the current root element as n:
        n = g.node(vid)
        if n.struct_mass > 0:
            # We define nitrogen active uptake from soil
            n.influx_N = (n.soil_N * vmax_N_root / (
                    n.soil_N + affinity_N_root)) * (2 * np.pi * n.radius * n.length)

            # We define mass root N concentration
            N_concentration = n.N/n.struct_mass
            # We define active xylem loading from root segment
            n.loading_N = (N_concentration * vmax_N_xylem / (
                    N_concentration + affinity_N_xylem)) * (2 * np.pi * n.radius * xylem_to_root * n.length)

            n.N += time_step * (n.influx_N - n.loading_N)
            #print(n.influx_N, n.loading_N, n.N)


    return g

def metabolism_N(g, **kwds):
    return g