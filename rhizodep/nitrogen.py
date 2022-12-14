from openalea.mtg import *
from openalea.mtg.traversal import post_order
import pickle
import numpy as np


def init_N(g,
           soil_N: float = 0.1,
           N: float = 0.1,
           xylem_N: float = 0.1,
           xylem_volume: float = 0,
           influx_N: float = 0,
           loading_N: float = 0):
    """
    Description
    Initialization of nitrogen-related variables

    Parameters
    :param g: MTG (dict)
    :param N: Local nitrogen content (mol)
    :param xylem_N: Global xylem nitrogen content (mol)
    :param xylem_volume: Global xylem vessel volume (m3)
    :param influx_N: Local nitrogen influx from soil
    :param loading_N: Local nitrogen loading to xylem

    Hypothesis
    H1 :
    H2 :


    """
    # Variable initialisation in MTG
    keywords = dict(soil_N=soil_N,
                    N=N,
                    influx_N=influx_N,
                    loading_N=loading_N)
    props = g.properties()
    for name in keywords:
        props.setdefault(name, {})

    max_scale = g.max_scale()
    for vid in g.vertices(scale=max_scale):
        for name, value in keywords.items():
            props[name][vid] = value

    # global vessel's property initialisation in first node
    g.node(0).xylem_N = xylem_N
    g.node(0).xylem_volume = xylem_volume

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
                xylem_to_root : float = 0.2
                ):
    '''
    Description
    ___________
    Nitrogen transport between local soil, local root segment and global vessels (xylem and phloem).

    :param g: MTG (dict)
    :param affinity_N_root: Active transport from soil Km parameter (mol.m-3)
    :param vmax_N_root: Surfacic maximal active transport rate to root (mol.m-2.s-1)
    :param affinity_N_xylem: Active transport from root Km parameter (mol.g-1)
    :param vmax_N_xylem: Surfacic maximal active transport rate to xylem (mol.m-2.s-1)
    :param xylem_to_root: Radius ratio between mean xylem and root segment (adim)

    Hypothesis
    H1: We summarize radial active transport controls (transporter density, affinity regulated with genetics
    or environnemental control, etc) as one mean transporter following Michaelis Menten's model.

    H2: We can summarize apoplastic and symplastic radial transport through one radial transport.
    Differentiation with epidermis conductance loss, root hair density, aerenchyma, etc, is supposed to impact Vmax.
    '''
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

            #print(n.influx_N, n.loading_N)


    return g

def metabolism_N(g, **kwds):
    return g

def update_N(g,
             xylem_to_root = 0.2,
             time_step = 3600):

    # Volume reinitialisation for update
    g.node(0).xylem_volume = 0

    # No order in update propagation
    max_scale = g.max_scale()
    for vid in g.vertices(scale=max_scale):
        # We define current root element as vid
        n = g.node(vid)

        # Local nitrogen pool update
        n.N += time_step * (n.influx_N - n.loading_N)

        # Global vessel's nitrogen pool update
        g.node(0).xylem_N += n.loading_N
        g.node(0).xylem_volume += np.pi * n.length * (n.radius*xylem_to_root)**2

    return g