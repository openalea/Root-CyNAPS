"""
rhizodep.topology
_________________
This is the radial root segment topology module for rhizodep exchange surfaces.

Documentation and features
__________________________
This module is meant to account for root's exchange surface plasticity and to estimate the surface proportion involved in these exchanges 
relative to boundary tissues's differenciation (xylem, endodermis, epidermis).
Apoplastic parietal resistance is though to be accounted for in these differentiations.

Main functions
______________
Dataclasses are used for MTG properties' initialization and parametrization.
Classes' names represent accounted hypothesis in the progressive development of the model.

Methods :
- update_topology() :   This method first compute boundary tissues differenciations relative to the segment structural mass.
                        Then, it updates the active exchange surfaces accounting for these differenciation processes.
                        For now, tissue surfaces relative to segment radius are supposed constant. This might be later questionned by litterature.
"""

# Imports

import numpy as np
from dataclasses import dataclass


# Dataclass for initialisation and parametrization.

# Properties' init

@dataclass
class InitSurfaces:
    root_exchange_surface: float = 0    # (m-2)
    stele_exchange_surface: float = 0   # (m-2)
    apoplasmic_stele: float = 0     # (adim)

# Parameters' default value

@dataclass
class TissueTopology:
    begin_xylem_diff: float = 0     # (g) structural mass at which xylem differentiation begins
    span_xylem_diff: float = 0.01    # (g) structural mass range width during which xylem differentiation occcurs
    endodermis_diff_rate: float = 2     # (g-1) endodermis suberisation rate
    epidermis_diff_rate: float = 1      # (g-1) epidermis suberisation rate
    epidermis_ratio: float = 4  # (adim) epidermis surface ratio over root's cylinder surface
    cortex_ratio: float = 30    # (adim) cortex surface ratio over root's cylinder surface
    stele_ratio: float = 20     # (adim) stele + endodermis surface ratio over root's cylinder surface



class RadialTopology:
    def __init__(self, g, root_exchange_surface, stele_exchange_surface, apoplasmic_stele):

        # New properties' creation in MTG
        keywords = dict(
        root_exchange_surface=root_exchange_surface,
        stele_exchange_surface=stele_exchange_surface,
        apoplasmic_stele=apoplasmic_stele)

        props = g.properties()
        for name in keywords:
            props.setdefault(name, {})

        # vertices storage for future calls in for loops
        self.vertices = g.vertices(scale=g.max_scale())
        for vid in self.vertices:
            for name, value in keywords.items():
                # Effectively creates the new property
                props[name][vid] = value
        
        # Accessing properties once, pointing to g for further modifications
        states = """
                        root_exchange_surface
                        stele_exchange_surface
                        apoplasmic_stele
                        radius
                        living_root_hairs_external_surface
                        struct_mass
                        """.split()
        
        for name in states:
            setattr(self, name, props[name])

    def update_topology(self, begin_xylem_diff, span_xylem_diff, endodermis_diff_rate, epidermis_diff_rate, epidermis_ratio, cortex_ratio, stele_ratio):
        """
        Description :
        This method first compute boundary tissues differenciations relative to the segment structural mass.
        Then, it updates the active exchange surfaces accounting for these differenciation processes.
        For now, tissue surfaces relative to segment radius are supposed constant. This might be later questionned by litterature.

        Parameters :
        :param begin_xylem_diff: (g) structural mass at which xylem differentiation begins
        :param span_xylem_diff: (g) structural mass range width during which xylem differentiation occcurs
        :param endodermis_diff_rate: (g-1) endodermis suberisation rate
        :param epidermis_diff_rate: (g-1) epidermis suberisation rate
        :param epidermis_ratio: (adim) epidermis surface ratio over root's cylinder surface
        :param cortex_ratio: (adim) cortex surface ratio over root's cylinder surface
        :param stele_ratio: (adim) stele + endodermis surface ratio over root's cylinder surface

        Hypothesis :
        H1 : xylem differentiation before endordermis differentiation opens a soil-xylem apoplasmic pathway.
        H2 : progressive endodermis then epidermis differentiation lowers the active exchange surface between soil and symplasmic parenchyma. 
        Apoplasmic parietal resistance is also accounted for in this formalism.
        H3 : for now, tissue surfaces are supposed linearly proportional to segment radius. This might be plasticised in further developments.
        """
        # for all root segments in MTG...
        for vid in self.vertices:

            # if root segment emerged
            if self.struct_mass[vid] > 0:
        
                # Update boundary layers' differenciation
                precision = 0.99

                xylem_differentiation = 1 / (1 + (precision/((1-precision) * np.exp(-begin_xylem_diff))
                        * np.exp(-self.struct_mass[vid] / span_xylem_diff)))
                endodermis_differentiation = np.exp(-endodermis_diff_rate * self.struct_mass[vid])
                epidermis_differentiation = np.exp(-epidermis_diff_rate * self.struct_mass[vid])
                
                # Update exchange surfaces

                # Exchanges between soil and symplasmic parenchyma
                self.root_exchange_surface[vid] = 2 * np.pi * self.radius[vid] *(
                    epidermis_ratio +
                    cortex_ratio * epidermis_differentiation +
                    stele_ratio * endodermis_differentiation
                ) + self.living_root_hairs_external_surface[vid]

                # Exchanges between symplamic parenchyma and xylem
                self.stele_exchange_surface[vid] = 2 * np.pi * self.radius[vid] * stele_ratio * xylem_differentiation

                # Apoplasmic exchanges factor between soil and xylem
                self.apoplasmic_stele[vid] = xylem_differentiation * endodermis_differentiation

        
