"""
rhizodep.nitrogen
_________________
Root nitrogen cycle model
"""

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
    :param soil_N: Local soil nitrogen concentration (mol.m-3)
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

    # props['xylem_N']={0:xylem_N}
    # props['xylem_volume']={0:xylem_volume}
    plant = g.node(0)
    plant.xylem_N = xylem_N
    plant.xylem_volume = xylem_volume

    return g


# Example of long calculation time related to repeating calls


def transport_N(g,
                affinity_N_root: float = 0.01,
                vmax_N_root: float = 0.5,
                affinity_N_xylem: float = 10,
                vmax_N_xylem: float = 0.1,
                xylem_to_root: float = 0.2
                ):
    """
    Description
    ___________
    Nitrogen transport between local soil, local root segment and global vessels (xylem and phloem).

    Parameters
    __________
    :param g: MTG (dict)
    :param affinity_N_root: Active transport from soil Km parameter (mol.m-3)
    :param vmax_N_root: Surfacic maximal active transport rate to root (mol.m-2.s-1)
    :param affinity_N_xylem: Active transport from root Km parameter (mol.g-1)
    :param vmax_N_xylem: Surfacic maximal active transport rate to xylem (mol.m-2.s-1)
    :param xylem_to_root: Radius ratio between mean xylem and root segment (adim)

    Hypothesis
    __________
    H1: We summarize radial active transport controls (transporter density, affinity regulated with genetics
    or environnemental control, etc) as one mean transporter following Michaelis Menten's model.

    H2: We can summarize apoplastic and symplastic radial transport through one radial transport.
    Differentiation with epidermis conductance loss, root hair density, aerenchyma, etc, is supposed to impact Vmax.
    """

    # Extract local properties once pointing to g
    props = g.properties()
    # N related
    soil_N = props['soil_N']
    N = props['N']
    influx_N = props['influx_N']
    loading_N = props['loading_N']
    # main model related
    length = props['length']
    radius = props['radius']
    struct_mass = props['struct_mass']

    # No order in update propagation
    max_scale = g.max_scale()
    for vid in g.vertices(scale=max_scale):
        # if root segment emerged
        if struct_mass[vid] > 0:
            # We define nitrogen active uptake from soil
            influx_N[vid] = (soil_N[vid] * vmax_N_root / (
                    soil_N[vid] + affinity_N_root)) * (2 * np.pi * radius[vid] * length[vid])

            # We define mass root N concentration
            N_concentration = N[vid] / struct_mass[vid]
            # We define active xylem loading from root segment
            loading_N[vid] = (N_concentration * vmax_N_xylem / (
                    N_concentration + affinity_N_xylem)) * (2 * np.pi * radius[vid] * xylem_to_root * length[vid])

            # print(influx_N[vid], loading_N[vid])

    return g


def metabolism_N(g):
    return g


def update_N(g,
             xylem_to_root=0.2,
             time_step=3600):
    # Extract plant-level properties once
    plant = g.node(0)
    xylem_volume = plant.xylem_volume = 0  # Recomputed
    xylem_N = plant.xylem_N

    # Extract local properties once, pointing to g
    props = g.properties()
    # N related
    N = props['N']
    influx_N = props['influx_N']
    loading_N = props['loading_N']
    # main model related
    length = props['length']
    radius = props['radius']

    # No order in update propagation
    max_scale = g.max_scale()
    for vid in g.vertices(scale=max_scale):
        # Local nitrogen pool update
        N[vid] += time_step * (influx_N[vid] - loading_N[vid])

        # Global vessel's nitrogen pool update
        xylem_N += loading_N[vid]
        xylem_volume += np.pi * length[vid] * (radius[vid] * xylem_to_root) ** 2

    # Update plant-level properties
    plant.xylem_N = xylem_N
    plant.xylem_volume = xylem_volume

    return g
