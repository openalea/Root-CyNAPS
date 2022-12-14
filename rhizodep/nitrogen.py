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
           soil_Nm: float = 0.1,
           Nm: float = 0.1,
           xylem_Nm: float = 0.1,
           xylem_volume: float = 0,
           influx_Nm: float = 0,
           loading_Nm: float = 0):
    """
    Description
    Initialization of nitrogen-related variables

    Parameters
    :param g: MTG (dict)
    :param soil_Nm: Local soil nitrogen volumic concentration (mol.m-3)
    :param Nm: Local nitrogen massic concentration (mol.g-1)
    :param xylem_Nm: Global xylem nitrogen volumic concentration (mol.m-3)
    :param xylem_volume: Global xylem vessel volume (m3)
    :param influx_Nm: Local nitrogen influx from soil (mol.s-1)
    :param loading_Nm: Local nitrogen loading to xylem (mol.s-1)

    Hypothesis
    H1 :
    H2 :


    """
    # Variable initialisation in MTG
    keywords = dict(soil_Nm=soil_Nm,
                    Nm=Nm,
                    influx_Nm=influx_Nm,
                    loading_Nm=loading_Nm)
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
    plant.xylem_Nm = xylem_Nm
    plant.xylem_volume = xylem_volume

    return g


# Example of long calculation time related to repeating calls


def transport_N(g,
                # kinetic parameters
                affinity_Nm_root: float = 0.01,
                vmax_Nm_emergence: float = 0.5,
                affinity_Nm_xylem: float = 10,
                # metabolism-related parameters
                transport_C_regulation: float = 1,
                # architecture parameters
                xylem_to_root: float = 0.2,
                epiderm_differentiation: float = 0.1,
                endoderm_differentiation: float = 0.1
                ):
    """
    Description
    ___________
    Nitrogen transport between local soil, local root segment and global vessels (xylem and phloem).

    Parameters
    __________
    :param g: MTG (dict)
    :param affinity_Nm_root: Active transport from soil Km parameter (mol.m-3)
    :param vmax_Nm_root: Surfacic maximal active transport rate to root (mol.m-2.s-1)
    :param affinity_Nm_xylem: Active transport from root Km parameter (mol.g-1)
    :param vmax_Nm_xylem: Surfacic maximal active transport rate to xylem (mol.m-2.s-1)
    :param transport_C_regulation:
    :param xylem_to_root: Radius ratio between mean xylem and root segment (adim)

    Hypothesis
    __________
    H1: We summarize radial active transport controls (transporter density, affinity regulated with genetics
    or environnemental control, etc) as one mean transporter following Michaelis Menten's model.

    H2: We can summarize apoplastic and symplastic radial transport through one radial transport.
    Differentiation with epidermis conductance loss, root hair density, aerenchyma, etc, is supposed to impact Vmax.

    H3: We declare similar kinetic parameters for soil-root and root-xylem active transport (exept for concentration conflict)
    """

    # Extract local properties once pointing to g
    props = g.properties()
    # N related
    soil_Nm = props['soil_Nm']
    Nm = props['Nm']
    influx_Nm = props['influx_Nm']
    loading_Nm = props['loading_Nm']
    # main model related
    length = props['length']
    radius = props['radius']
    struct_mass = props['struct_mass']
    C_hexose_root = props['C_hexose_root']
    thermal_time_since_emergence = props['thermal_time_since_emergence']

    # No order in update propagation
    max_scale = g.max_scale()
    for vid in g.vertices(scale=max_scale):
        # if root segment emerged
        if struct_mass[vid] > 0:
            # We define nitrogen active uptake from soil
            # Vmax supposed affected by root aging
            vmax_Nm_root = vmax_Nm_emergence * np.exp(- epiderm_differentiation * thermal_time_since_emergence[vid])
            # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
            influx_Nm[vid] = (soil_Nm[vid] * vmax_Nm_root / (soil_Nm[vid] + affinity_Nm_root)) \
                             * (2 * np.pi * radius[vid] * length[vid]) \
                             * (C_hexose_root[vid] / (C_hexose_root[vid] + transport_C_regulation))

            # We define active xylem loading from root segment
            # Vmax supposed affected by root aging
            vmax_Nm_xylem = vmax_Nm_emergence * np.exp(- endoderm_differentiation * thermal_time_since_emergence[vid])
            # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
            loading_Nm[vid] = (Nm[vid] * vmax_Nm_xylem / (Nm[vid] + affinity_Nm_xylem)) \
                              * (2 * np.pi * radius[vid] * xylem_to_root * length[vid]) \
                              * (C_hexose_root[vid] / (C_hexose_root[vid] + transport_C_regulation))

            # print(influx_N[vid], loading_N[vid])

    return g


def metabolism_N(g):
    return g


def update_N(g,
             xylem_to_root=0.2,
             time_step=3600):
    # Extract plant-level properties once
    plant = g.node(0)
    xylem_volume = plant.xylem_volume
    # We define xylem nitrogen content (mol) from previous volume and concentrations.
    xylem_Nm_content = plant.xylem_Nm * xylem_volume
    # Computing actualised volume
    xylem_volume = 0

    # Extract local properties once, pointing to g
    props = g.properties()
    # N related
    Nm = props['Nm']
    influx_Nm = props['influx_Nm']
    loading_Nm = props['loading_Nm']
    # main model related
    length = props['length']
    radius = props['radius']
    struct_mass = props['struct_mass']

    # No order in update propagation
    max_scale = g.max_scale()
    for vid in g.vertices(scale=max_scale):
        # if root segment emerged
        if struct_mass[vid] > 0:
            # Local nitrogen pool update
            Nm[vid] += (time_step / struct_mass[vid]) * (influx_Nm[vid] - loading_Nm[vid])

            # Global vessel's nitrogen pool update
            xylem_Nm_content += time_step * loading_Nm[vid]
            xylem_volume += np.pi * length[vid] * (radius[vid] * xylem_to_root) ** 2

    # Update plant-level properties
    plant.xylem_Nm = xylem_Nm_content / xylem_volume
    plant.xylem_volume = xylem_volume

    return g
