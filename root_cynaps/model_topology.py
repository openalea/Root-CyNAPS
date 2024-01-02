"""
root_cynaps.topology
_________________
This is the radial root segment topology module for root_cynaps exchange surfaces.

Documentation and features
__________________________
This module is meant to account for root's exchange surface plasticity and to estimate the surface proportion involved in these exchanges 
relative to boundary tissues' differentiation (xylem, endodermis, epidermis).
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
from dataclasses import dataclass, asdict


# Dataclass for initialisation and parametrization.

# Properties' init
# TODO merge with rhizodep
@dataclass
class InitSurfaces:
    root_exchange_surface: float = 0  # (m2)
    cortex_exchange_surface: float = 0  # (m2)
    apoplasmic_exchange_surface: float = 0  # (m2)
    stele_exchange_surface: float = 0  # (m2)
    phloem_exchange_surface: float = 0  # (m2)
    apoplasmic_stele: float = 0  # (adim)
    xylem_volume: float = 0  # (m3)


# Parameters' default value

@dataclass
class TissueTopology:
    begin_xylem_diff: float = 0  # (g) structural mass at which xylem differentiation begins
    span_xylem_diff: float = 2.8e-4  # (g) structural mass range width during which xylem differentiation occurs
    endodermis_diff_rate: float = 1.2e4  # (g-1) endodermis suberisation rate
    epidermis_diff_rate: float = 1.8e3  # (g-1) epidermis suberisation rate
    cortex_ratio: float = 29  # (adim) cortex (+epidermis) surface ratio over root's cylinder surface
    stele_ratio: float = 11  # (adim) stele (+endodermis) surface ratio over root's cylinder surface
    phloem_ratio: float = 2.5  # (adim) phloem surface ratio over root's cylinder surface
    xylem_cross_area_ratio: float = 0.84 * (0.36 ** 2)  # (adim) apoplasmic cross-section area ratio * stele radius ratio^2


class RadialTopology:
    def __init__(self, g, root_exchange_surface, cortex_exchange_surface, apoplasmic_exchange_surface, stele_exchange_surface,
                 phloem_exchange_surface, apoplasmic_stele, xylem_volume):

        self.g = g

        # New properties' creation in MTG
        self.keywords = dict(
            root_exchange_surface=root_exchange_surface,
            cortex_exchange_surface=cortex_exchange_surface,
            apoplasmic_exchange_surface=apoplasmic_exchange_surface,
            stele_exchange_surface=stele_exchange_surface,
            phloem_exchange_surface=phloem_exchange_surface,
            apoplasmic_stele=apoplasmic_stele,
            xylem_volume=xylem_volume)

        props = g.properties()
        for name in self.keywords:
            props.setdefault(name, {})

        # vertices storage for future calls in for loops
        self.vertices = g.vertices(scale=g.max_scale())
        for vid in self.vertices:
            for name, value in self.keywords.items():
                # Effectively creates the new property
                props[name][vid] = value

        # Accessing properties once, pointing to g for further modifications
        self.states = """
                        root_exchange_surface
                        cortex_exchange_surface
                        apoplasmic_exchange_surface
                        stele_exchange_surface
                        phloem_exchange_surface
                        apoplasmic_stele
                        xylem_volume
                        length
                        radius
                        struct_mass
                        """.split()
        # NO ROOT HAIR PROPERTY

        for name in self.states:
            setattr(self, name, props[name])

        # Collar geometry corrective
        self.length[1] = 0.003
        # self.radius[1] = 0.002

        self.update_topology(**asdict(TissueTopology()))

    def add_properties_to_new_segments(self):
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        for vid in self.vertices:
            if vid not in list(self.root_exchange_surface.keys()):
                for prop in list(self.keywords.keys()):
                    getattr(self, prop)[vid] = 0
        # WARNING? OPTIONAL AND TO REMOVE WHEN NO SIMULATION FROM FILE
        for name in self.states:
            setattr(self, name, self.g.properties()[name])

    def update_topology(self, begin_xylem_diff, span_xylem_diff, endodermis_diff_rate, epidermis_diff_rate,
                        cortex_ratio, stele_ratio, phloem_ratio, xylem_cross_area_ratio):
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
        :param cortex_ratio: (adim) cortex (+epidermis) surface ratio over root's cylinder surface
        :param stele_ratio: (adim) (adim) stele (+endodermis) surface ratio over root's cylinder surface
        :param phloem_ratio: (adim) phloem surface ratio over root's cylinder surface

        Hypothesis :
        H1 : xylem differentiation before endodermis differentiation opens a soil-xylem apoplasmic pathway.
        H2 : progressive endodermis then epidermis differentiation lowers the active exchange surface between soil and symplasmic parenchyma. 
        Apoplasmic parietal resistance is also accounted for in this formalism.
        H3 : for now, tissue surfaces are supposed linearly proportional to segment radius. This might be plasticised in further developments.
        """
        self.add_properties_to_new_segments()

        # for all root segments in MTG...
        for vid in self.vertices:

            # if root segment emerged
            if self.struct_mass[vid] > 0:
                # Update boundary layers' differentiation
                precision = 0.99

                xylem_differentiation = 1 / (1 + (precision / ((1 - precision) * np.exp(-begin_xylem_diff))
                                                  * np.exp(-self.struct_mass[vid] / span_xylem_diff)))

                endodermis_differentiation = np.exp(-endodermis_diff_rate * self.struct_mass[vid])

                epidermis_differentiation = np.exp(-epidermis_diff_rate * self.struct_mass[vid])

                # Update exchange surfaces

                # N exchanges between soil and symplasmic parenchyma
                self.root_exchange_surface[vid] = 2 * np.pi * self.radius[vid] * self.length[vid] * (
                        cortex_ratio * epidermis_differentiation +
                        stele_ratio * endodermis_differentiation
                )  # + self.root_hairs_external_surface[vid]
                #   NO ROOT HAIR PROPERTY

                # Water exchanges between soil and symplasmic cortex at equilibrium with stele

                self.cortex_exchange_surface[vid] = 2 * np.pi * self.radius[vid] * self.length[vid] * (
                        cortex_ratio * epidermis_differentiation)  # + self.root_hairs_external_surface[vid]
                # NO ROOT HAIR PROPERTY

                # Water exchanges between soil and xylem
                self.apoplasmic_exchange_surface[vid] = 2 * np.pi * self.radius[vid] * self.length[vid] * endodermis_differentiation

                # Exchanges between symplamic parenchyma and xylem
                self.stele_exchange_surface[vid] = 2 * np.pi * self.radius[vid] * self.length[
                    vid] * stele_ratio * xylem_differentiation

                # Phloem exchangee surface, accessible from start
                self.phloem_exchange_surface[vid] = 2 * np.pi * self.radius[vid] * self.length[vid] * phloem_ratio

                # Apoplasmic exchanges factor between soil and xylem
                self.apoplasmic_stele[vid] = xylem_differentiation * endodermis_differentiation

                # Conductive apoplasmic volume

                self.xylem_volume[vid] = np.pi * (self.radius[vid] ** 2) * xylem_cross_area_ratio * self.length[vid]

