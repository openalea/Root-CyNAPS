"""
rhizodep.nitrogen
_________________
This is the main nitrogen cycle module for roots

Documentation and features
__________________________

Main functions
______________
Classes' names represent accounted hypothesis in the progressive development of the model.
Methods' names are systematic through all class for ease of use :

TODO : report functions descriptions
init_N()
transport_N()
metabolism_N()
update_N()

Use examples
____________

With carbon model :

"""

import numpy as np
from openalea.mtg.aml import VtxList, Father, Scale


class GlobalVessels:

    def __init__(self, g, Nm, AA, influx_Nm, diffusion_AA_soil, loading_Nm, loading_AA, diffusion_Nm_phloem,
                 diffusion_AA_phloem, AA_synthesis, struct_synthesis, storage_synthesis, AA_catabolism,
                 storage_catabolism, xylem_Nm, xylem_AA, xylem_struct_mass, phloem_Nm, phloem_AA,
                 phloem_struct_mass, Nm_root_shoot_xylem, AA_root_shoot_xylem, Nm_root_shoot_phloem, AA_root_shoot_phloem):

        """
        Description
        Initialization of nitrogen-related variables

        Parameters
        :param g: MTG
        :param Nm: Local mineral nitrogen volumic concentration (mol.m-3)
        :param influx_Nm: Local mineral nitrogen influx from soil (mol.s-1)
        :param loading_Nm: Local mineral nitrogen loading to xylem (mol.s-1)
        :param diffusion_Nm_phloem: Local mineral nitrogen diffusion between cortex and phloem (mol.s-1)
        :param xylem_Nm: Global xylem mineral nitrogen volumic concentration (mol.m-3)
        :param xylem_volume: Global xylem vessel volume (m3)
        :param phloem_Nm: Global phloem mineral nitrogen volumic concentration (mol.m-3)
        :param phloem_volume: Global phloem vessel volume (m3)
        :param Nm_root_shoot_xylem: Mineral nitrogen transport to shoot from root xylem (mol.s-1)
        :param Nm_root_shoot_phloem: Mineral nitrogen transport from shoot to root phloem (mol.s-1)

        Hypothesis
        H1 :
        H2 :
        """

        self.g = g
        # New properties' creation in MTG
        keywords = dict(Nm=Nm,
                        AA=AA,
                        influx_Nm=influx_Nm,
                        diffusion_AA_soil=diffusion_AA_soil,
                        loading_Nm=loading_Nm,
                        loading_AA=loading_AA,
                        diffusion_Nm_phloem=diffusion_Nm_phloem,
                        diffusion_AA_phloem=diffusion_AA_phloem,
                        AA_synthesis=AA_synthesis,
                        struct_synthesis=struct_synthesis,
                        storage_synthesis=storage_synthesis,
                        AA_catabolism=AA_catabolism,
                        storage_catabolism=storage_catabolism,
                        xylem_Nm=xylem_Nm,
                        xylem_AA=xylem_AA,
                        xylem_struct_mass=xylem_struct_mass,
                        phloem_Nm=phloem_Nm,
                        phloem_AA=phloem_AA,
                        phloem_struct_mass=phloem_struct_mass,
                        Nm_root_shoot_xylem=Nm_root_shoot_xylem,
                        AA_root_shoot_xylem=AA_root_shoot_xylem,
                        Nm_root_shoot_phloem=Nm_root_shoot_phloem,
                        AA_root_shoot_phloem=AA_root_shoot_phloem
                        )

        props = self.g.properties()
        for name in keywords:
            props.setdefault(name, {})

        # vertices storage for future calls in for loops
        self.vertices = self.g.vertices(scale=g.max_scale())
        for vid in self.vertices:
            for name, value in keywords.items():
                # Effectively creates the new property
                props[name][vid] = value

        # Accessing properties once, pointing to g for further modifications
        # N related
        # main model related
        states = """
                soil_Nm
                soil_AA
                Nm
                AA
                volume
                influx_Nm
                diffusion_AA_soil
                loading_Nm
                loading_AA
                diffusion_Nm_phloem
                diffusion_AA_phloem
                AA_synthesis
                struct_synthesis
                storage_synthesis
                AA_catabolism
                storage_catabolism
                xylem_Nm
                xylem_AA
                xylem_struct_mass
                phloem_Nm
                phloem_AA
                phloem_struct_mass
                Nm_root_shoot_xylem
                AA_root_shoot_xylem
                Nm_root_shoot_phloem
                AA_root_shoot_phloem
                length
                radius
                struct_mass
                C_hexose_root
                C_hexose_reserve
                thermal_time_since_emergence
                """.split()

        for name in states:
            setattr(self, name, props[name])

        # Note : Global properties are declared as local ones, but only vertice 1 will be updated

    def transport_N(self, affinity_Nm_root, vmax_Nm_emergence, affinity_Nm_xylem, vmax_AA_emergence, affinity_AA_xylem,
                    diffusion_phloem, diffusion_soil, transport_C_regulation, transport_N_regulation, xylem_to_root, phloem_to_root, epiderm_differentiation, endoderm_differentiation):

        """
        Description
        ___________
        Nitrogen transport between local soil, local root segment and global vessels (xylem and phloem).

        Parameters
        __________
        :param affinity_Nm_root: Active transport from soil Km parameter (mol.m-3)
        :param vmax_Nm_emergence: Surfacic maximal active transport rate in roots (mol.m-2.s-1)
        :param affinity_Nm_xylem: Active transport from root Km parameter (mol.m-3)
        :param diffusion_phloem: Mineral nitrogen diffusion parameter (m.s-1)
        :param transport_C_regulation: Affinity coefficient for the nitrogen active transport regulation function
        by root C (mol.g-1) (?)
        :param transport_N_regulation: Affinity coefficient for the nitrogen active transport regulation function
        by root mineral nitrogen (mol.m-3)
        :param xylem_to_root: Radius ratio between mean xylem and root segment (adim)
        :param phloem_to_root: Radius ratio between mean phloem and root segment (adim)
        :param epiderm_differentiation: Epiderm differentiation rate (°C-1.s-1)
        :param endoderm_differentiation: Endoderm differentiation rate (°C-1.s-1)

        Hypothesis
        __________
        H1: We summarize radial active transport controls (transporter density, affinity regulated with genetics
        or environmental control, etc) as one mean transporter following Michaelis Menten's model.

        H2: We can summarize apoplastic and symplastic radial transport through one radial transport.
        Differentiation with epidermis conductance loss, root hair density, aerenchyma, etc, is supposed to impact Vmax.

        H3: We declare similar kinetic parameters for soil-root and root-xylem active transport (exept for concentration conflict)
        """

        # No order in update propagation
        for vid in self.vertices:

            # if root segment emerged
            if self.struct_mass[vid] > 0:

                # MINERAL NITROGEN TRANSPORT

                # We define mineral nitrogen active uptake from soil
                # Vmax supposed affected by root aging
                vmax_Nm_root = vmax_Nm_emergence * np.exp(
                    - epiderm_differentiation * self.thermal_time_since_emergence[vid])

                # Km is supposed affected by different processes regulated by destination nitrogen availability
                # (HATS/LATS composition and availability, phosphorylation, etc)
                km_Nm_root = affinity_Nm_root * np.exp(transport_N_regulation * self.Nm[vid])

                # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
                self.influx_Nm[vid] = ((self.soil_Nm[vid] * vmax_Nm_root / (self.soil_Nm[vid] + km_Nm_root))
                                       * (2 * np.pi * self.radius[vid] * self.length[vid])
                                       * (self.C_hexose_root[vid] / (self.C_hexose_root[vid] + transport_C_regulation)))

                # We define active xylem loading from root segment
                # Vmax supposed affected by root aging
                vmax_Nm_xylem = vmax_Nm_emergence * np.exp(
                    - endoderm_differentiation * self.thermal_time_since_emergence[vid])

                # Km is defined as a constant here because xylem is global
                # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
                self.loading_Nm[vid] = ((self.Nm[vid] * vmax_Nm_xylem / (self.Nm[vid] + affinity_Nm_xylem))
                                        * (2 * np.pi * self.radius[vid] * xylem_to_root * self.length[vid])
                                        * (self.C_hexose_root[vid] / (
                                    self.C_hexose_root[vid] + transport_C_regulation)))

                # Passive radial diffusion between phloem and cortex through plasmodesmata
                self.diffusion_Nm_phloem[vid] = (diffusion_phloem * (self.phloem_Nm[1] - self.Nm[vid])
                                                * (2 * np.pi * self.radius[vid] * phloem_to_root * self.length[vid]))

                # AMINO ACID TRANSPORT

                # We define amino acid passive diffusion to soil
                self.diffusion_AA_soil[vid] = (diffusion_soil * (self.AA[vid] - self.soil_AA[vid])
                                                 * (2 * np.pi * self.radius[vid] * phloem_to_root * self.length[vid]))

                # We define active xylem loading from root segment
                # Vmax supposed affected by root aging
                vmax_AA_xylem = vmax_AA_emergence * np.exp(
                    - endoderm_differentiation * self.thermal_time_since_emergence[vid])

                # Km is defined as a constant here because xylem is global
                # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
                self.loading_AA[vid] = ((self.AA[vid] * vmax_AA_xylem / (self.AA[vid] + affinity_AA_xylem))
                                        * (2 * np.pi * self.radius[vid] * xylem_to_root * self.length[vid])
                                        * (self.C_hexose_root[vid] / (
                                self.C_hexose_root[vid] + transport_C_regulation)))

                # Passive radial diffusion between phloem and cortex through plasmodesmata
                self.diffusion_AA_phloem[vid] = (diffusion_phloem * (self.phloem_AA[1] - self.AA[vid])
                                                 * (2 * np.pi * self.radius[vid] * phloem_to_root * self.length[vid]))

    def metabolism_N(self, smax_AA, affinity_Nm_AA, affinity_C_AA, smax_struct, affinity_AA_struct, smax_stor,
                     affinity_AA_stor, cmax_stor, affinity_stor_catab, cmax_AA, affinity_AA_catab, storage_C_regulation):

        """
        Description
        ___________
        Nitrogen metabolism within local root segment cortex

        Parameters
        __________
        :param smax_struct : Maximal organic structure synthesis from mineral nitrogen and labil C (mol.s-1)
        :param affinity_Nm_struct :
        :param affinity_C_struct :

        """
        # No order in update propagation
        for vid in self.vertices:

            # if root segment emerged
            if self.struct_mass[vid] > 0:
                # amino acid synthesis
                self.AA_synthesis[vid] = smax_AA / (
                    ((1 + affinity_Nm_AA) / self.Nm[vid])
                    + ((1 + affinity_C_AA) / self.C_hexose_root[vid])
                )

                # Organic structure synthesis
                self.struct_synthesis[vid] = (smax_struct * self.AA[vid]
                                               / (affinity_AA_struct + self.AA[vid]))

                # Organic storage synthesis (Michaelis-Menten kinetic)
                self.storage_synthesis[vid] = (smax_stor * self.AA[vid]
                                               /(affinity_AA_stor + self.AA[vid]))

                # Organic storage catabolism
                Km_stor_root = affinity_stor_catab * np.exp(storage_C_regulation * self.C_hexose_root[vid])
                self.storage_catabolism[vid] = cmax_stor * self.C_hexose_reserve[vid] / (
                        Km_stor_root + self.C_hexose_reserve[vid])

                # AA catabolism
                Km_stor_root = affinity_AA_catab * np.exp(storage_C_regulation * self.C_hexose_root[vid])
                self.AA_catabolism[vid] = cmax_AA * self.AA[vid] / (
                        Km_stor_root + self.AA[vid])

    def update_N(self, r_Nm_AA, r_AA_struct, r_AA_stor, xylem_to_root, phloem_to_root, time_step):
        """
        Description
        ___________

        Parameters
        __________
        :param r_Nm_struct : Nm mol consumed per mol of organic structural synthesis (adim)

        """

        # Computing vessels' mass as a fraction of total segments mass
        self.xylem_struct_mass[1] = sum(self.struct_mass.values()) * xylem_to_root
        self.phloem_struct_mass[1] = sum(self.struct_mass.values()) * phloem_to_root

        # No order in update propagation
        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:
                # Local volume update
                Nm_content = self.Nm[vid] * self.struct_mass[vid]
                AA_content = self.AA[vid] * self.struct_mass[vid]

                # Local nitrogen concentration update
                self.Nm[vid] += time_step / self.struct_mass[vid] * (
                        self.influx_Nm[vid]
                        + self.diffusion_Nm_phloem[vid]
                        - self.loading_Nm[vid]
                        - self.AA_synthesis[vid] * r_Nm_AA
                        + self.AA_catabolism[vid] / r_Nm_AA)

                self.AA[vid] += time_step / self.struct_mass[vid] * (
                        self.diffusion_AA_phloem[vid]
                        - self.diffusion_AA_soil[vid]
                        - self.loading_AA[vid]
                        + self.AA_synthesis[vid]
                        - self.struct_synthesis[vid] * r_AA_struct
                        - self.storage_synthesis[vid] * r_AA_stor
                        + self.storage_catabolism[vid] / r_AA_stor
                        - self.AA_catabolism[vid]
                        )

                # Global vessel's nitrogen pool update
                self.xylem_Nm[1] += time_step * self.loading_Nm[vid] / self.xylem_struct_mass[1]
                self.xylem_AA[1] += time_step * self.loading_AA[vid] / self.xylem_struct_mass[1]
                self.phloem_Nm[1] -= time_step * self.diffusion_Nm_phloem[vid] / self.xylem_struct_mass[1]
                self.phloem_AA[1] -= time_step * self.diffusion_AA_phloem[vid] / self.xylem_struct_mass[1]

        # Update plant-level properties
        self.xylem_Nm[1] -= time_step * self.Nm_root_shoot_xylem[1] / self.xylem_struct_mass[1]
        self.xylem_AA[1] -= time_step * self.AA_root_shoot_xylem[1] / self.xylem_struct_mass[1]
        self.phloem_Nm[1] += time_step * self.Nm_root_shoot_phloem[1] / self.phloem_struct_mass[1]
        self.phloem_AA[1] += time_step * self.AA_root_shoot_phloem[1] / self.phloem_struct_mass[1]


class DiscreteVessels:

    def __init__(self, g, Nm, AA, influx_Nm, diffusion_AA_soil, loading_Nm, loading_AA, diffusion_Nm_phloem,
                 diffusion_AA_phloem, axial_diffusion_Nm_xylem, axial_diffusion_AA_xylem, axial_diffusion_Nm_phloem,
                 axial_diffusion_AA_phloem, AA_synthesis, struct_synthesis, storage_synthesis, AA_catabolism,
                 storage_catabolism, xylem_Nm, xylem_AA, xylem_struct_mass, phloem_Nm, phloem_AA,
                 phloem_struct_mass, Nm_root_shoot_xylem, AA_root_shoot_xylem, Nm_root_shoot_phloem, AA_root_shoot_phloem):

        """
        Description
        Initialization of nitrogen-related variables

        Parameters
        :param g: MTG
        :param Nm: Local mineral nitrogen volumic concentration (mol.m-3)
        :param influx_Nm: Local mineral nitrogen influx from soil (mol.s-1)
        :param loading_Nm: Local mineral nitrogen loading to xylem (mol.s-1)
        :param diffusion_Nm_phloem: Local mineral nitrogen diffusion between cortex and phloem (mol.s-1)
        :param xylem_Nm: Global xylem mineral nitrogen volumic concentration (mol.m-3)
        :param xylem_volume: Global xylem vessel volume (m3)
        :param phloem_Nm: Global phloem mineral nitrogen volumic concentration (mol.m-3)
        :param phloem_volume: Global phloem vessel volume (m3)
        :param Nm_root_shoot_xylem: Mineral nitrogen transport to shoot from root xylem (mol.s-1)
        :param Nm_root_shoot_phloem: Mineral nitrogen transport from shoot to root phloem (mol.s-1)

        Hypothesis
        H1 :
        H2 :
        """

        self.g = g
        # New properties' creation in MTG
        keywords = dict(Nm=Nm,
                        AA=AA,
                        influx_Nm=influx_Nm,
                        diffusion_AA_soil=diffusion_AA_soil,
                        loading_Nm=loading_Nm,
                        loading_AA=loading_AA,
                        diffusion_Nm_phloem=diffusion_Nm_phloem,
                        diffusion_AA_phloem=diffusion_AA_phloem,
                        axial_diffusion_Nm_xylem=axial_diffusion_Nm_xylem,
                        axial_diffusion_AA_xylem=axial_diffusion_AA_xylem,
                        axial_diffusion_Nm_phloem=axial_diffusion_Nm_phloem,
                        axial_diffusion_AA_phloem=axial_diffusion_AA_phloem,
                        AA_synthesis=AA_synthesis,
                        struct_synthesis=struct_synthesis,
                        storage_synthesis=storage_synthesis,
                        AA_catabolism=AA_catabolism,
                        storage_catabolism=storage_catabolism,
                        xylem_Nm=xylem_Nm,
                        xylem_AA=xylem_AA,
                        xylem_struct_mass=xylem_struct_mass,
                        phloem_Nm=phloem_Nm,
                        phloem_AA=phloem_AA,
                        phloem_struct_mass=phloem_struct_mass,
                        Nm_root_shoot_xylem=Nm_root_shoot_xylem,
                        AA_root_shoot_xylem=AA_root_shoot_xylem,
                        Nm_root_shoot_phloem=Nm_root_shoot_phloem,
                        AA_root_shoot_phloem=AA_root_shoot_phloem
                        )

        props = self.g.properties()
        for name in keywords:
            props.setdefault(name, {})

        # vertices storage for future calls in for loops
        self.vertices = self.g.vertices(scale=g.max_scale())
        for vid in self.vertices:
            for name, value in keywords.items():
                # Effectively creates the new property
                props[name][vid] = value

        # Accessing properties once, pointing to g for further modifications
        # N related
        # main model related
        states = """
                soil_Nm
                soil_AA
                Nm
                AA
                volume
                influx_Nm
                diffusion_AA_soil
                loading_Nm
                loading_AA
                diffusion_Nm_phloem
                diffusion_AA_phloem
                axial_diffusion_Nm_xylem
                axial_diffusion_AA_xylem
                axial_diffusion_Nm_phloem
                axial_diffusion_AA_phloem
                AA_synthesis
                struct_synthesis
                storage_synthesis
                AA_catabolism
                storage_catabolism
                xylem_Nm
                xylem_AA
                xylem_struct_mass
                phloem_Nm
                phloem_AA
                phloem_struct_mass
                Nm_root_shoot_xylem
                AA_root_shoot_xylem
                Nm_root_shoot_phloem
                AA_root_shoot_phloem
                length
                radius
                struct_mass
                C_hexose_root
                C_hexose_reserve
                thermal_time_since_emergence
                """.split()

        for name in states:
            setattr(self, name, props[name])

        # Note : Global properties are declared as local ones, but only vertice 1 will be updated

    def transport_N(self, affinity_Nm_root, vmax_Nm_emergence, affinity_Nm_xylem, vmax_AA_emergence, affinity_AA_xylem,
                    diffusion_phloem, diffusion_soil, axial_diffusion_xylem, axial_diffusion_phloem,
                    transport_C_regulation, transport_N_regulation, xylem_to_root, phloem_to_root,
                    epiderm_differentiation, endoderm_differentiation):

        """
        Description
        ___________
        Nitrogen transport between local soil, local root segment and global vessels (xylem and phloem).

        Parameters
        __________
        :param affinity_Nm_root: Active transport from soil Km parameter (mol.m-3)
        :param vmax_Nm_emergence: Surfacic maximal active transport rate in roots (mol.m-2.s-1)
        :param affinity_Nm_xylem: Active transport from root Km parameter (mol.m-3)
        :param diffusion_phloem: Mineral nitrogen diffusion parameter (m.s-1)
        :param transport_C_regulation: Affinity coefficient for the nitrogen active transport regulation function
        by root C (mol.g-1) (?)
        :param transport_N_regulation: Affinity coefficient for the nitrogen active transport regulation function
        by root mineral nitrogen (mol.m-3)
        :param xylem_to_root: Radius ratio between mean xylem and root segment (adim)
        :param phloem_to_root: Radius ratio between mean phloem and root segment (adim)
        :param epiderm_differentiation: Epiderm differentiation rate (°C-1.s-1)
        :param endoderm_differentiation: Endoderm differentiation rate (°C-1.s-1)

        Hypothesis
        __________
        H1: We summarize radial active transport controls (transporter density, affinity regulated with genetics
        or environmental control, etc) as one mean transporter following Michaelis Menten's model.

        H2: We can summarize apoplastic and symplastic radial transport through one radial transport.
        Differentiation with epidermis conductance loss, root hair density, aerenchyma, etc, is supposed to impact Vmax.

        H3: We declare similar kinetic parameters for soil-root and root-xylem active transport (exept for concentration conflict)
        """

        # No order in update propagation
        for vid in self.vertices:

            # if root segment emerged
            if self.struct_mass[vid] > 0:

                ## RADIAL TRANSPORT

                # MINERAL NITROGEN TRANSPORT

                # We define mineral nitrogen active uptake from soil
                # Vmax supposed affected by root aging
                vmax_Nm_root = vmax_Nm_emergence * np.exp(
                    - epiderm_differentiation * self.thermal_time_since_emergence[vid])

                # Km is supposed affected by different processes regulated by destination nitrogen availability
                # (HATS/LATS composition and availability, phosphorylation, etc)
                km_Nm_root = affinity_Nm_root * np.exp(transport_N_regulation * self.Nm[vid])

                # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
                self.influx_Nm[vid] = ((self.soil_Nm[vid] * vmax_Nm_root / (self.soil_Nm[vid] + km_Nm_root))
                                       * (2 * np.pi * self.radius[vid] * self.length[vid])
                                       * (self.C_hexose_root[vid] / (self.C_hexose_root[vid] + transport_C_regulation)))

                # We define active xylem loading from root segment
                # Vmax supposed affected by root aging
                vmax_Nm_xylem = vmax_Nm_emergence * np.exp(
                    - endoderm_differentiation * self.thermal_time_since_emergence[vid])

                # Km is defined as a constant here because xylem is global
                # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
                self.loading_Nm[vid] = ((self.Nm[vid] * vmax_Nm_xylem / (self.Nm[vid] + affinity_Nm_xylem))
                                        * (2 * np.pi * self.radius[vid] * xylem_to_root * self.length[vid])
                                        * (self.C_hexose_root[vid] / (
                                    self.C_hexose_root[vid] + transport_C_regulation)))

                # Passive radial diffusion between phloem and cortex through plasmodesmata
                self.diffusion_Nm_phloem[vid] = (diffusion_phloem * (self.phloem_Nm[vid] - self.Nm[vid])
                                                * (2 * np.pi * self.radius[vid] * phloem_to_root * self.length[vid]))

                # AMINO ACID TRANSPORT

                # We define amino acid passive diffusion to soil
                self.diffusion_AA_soil[vid] = (diffusion_soil * (self.AA[vid] - self.soil_AA[vid])
                                                 * (2 * np.pi * self.radius[vid] * phloem_to_root * self.length[vid]))

                # We define active xylem loading from root segment
                # Vmax supposed affected by root aging
                vmax_AA_xylem = vmax_AA_emergence * np.exp(
                    - endoderm_differentiation * self.thermal_time_since_emergence[vid])

                # Km is defined as a constant here because xylem is global
                # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
                self.loading_AA[vid] = ((self.AA[vid] * vmax_AA_xylem / (self.AA[vid] + affinity_AA_xylem))
                                        * (2 * np.pi * self.radius[vid] * xylem_to_root * self.length[vid])
                                        * (self.C_hexose_root[vid] / (
                                self.C_hexose_root[vid] + transport_C_regulation)))

                # Passive radial diffusion between phloem and cortex through plasmodesmata
                self.diffusion_AA_phloem[vid] = (diffusion_phloem * (self.phloem_AA[vid] - self.AA[vid])
                                                 * (2 * np.pi * self.radius[vid] * phloem_to_root * self.length[vid]))

                ## AXIAL TRANSPORT

                neighbor = [self.g.parent(vid)] + self.g.children(vid)
                if None in neighbor:
                    neighbor.remove(None)

                # Reinitialization before computing for each neighbor
                self.axial_diffusion_Nm_xylem[vid] = 0.0
                self.axial_diffusion_Nm_phloem[vid] = 0.0
                self.axial_diffusion_AA_xylem[vid] = 0.0
                self.axial_diffusion_AA_phloem[vid] = 0.0

                for k in neighbor:
                    if self.struct_mass[k] > 0:
                        # MINERAL NITROGEN TRANSPORT
                        self.axial_diffusion_Nm_xylem[vid] += axial_diffusion_xylem * (self.xylem_Nm[k] - self.xylem_Nm[vid]) * (
                                                            np.pi * (xylem_to_root * (self.radius[vid] + self.radius[k]) / 2)**2)

                        self.axial_diffusion_Nm_phloem[vid] += axial_diffusion_phloem * (self.phloem_Nm[k] - self.phloem_Nm[vid]) * (
                                                            np.pi * (phloem_to_root * (self.radius[vid] + self.radius[k]) / 2) ** 2)

                        # AMINO ACID TRANSPORT
                        self.axial_diffusion_AA_xylem[vid] += axial_diffusion_xylem * (self.xylem_AA[k] - self.xylem_AA[vid]) * (
                                                            np.pi * (xylem_to_root * (self.radius[vid] + self.radius[k]) / 2) ** 2)

                        self.axial_diffusion_AA_phloem[vid] += axial_diffusion_phloem * (self.phloem_AA[k] - self.phloem_AA[vid]) * (
                                                            np.pi * (xylem_to_root * (self.radius[vid] + self.radius[k]) / 2) ** 2)
                print(self.axial_diffusion_Nm_xylem[vid])

    def metabolism_N(self, smax_AA, affinity_Nm_AA, affinity_C_AA, smax_struct, affinity_AA_struct, smax_stor,
                     affinity_AA_stor, cmax_stor, affinity_stor_catab, cmax_AA, affinity_AA_catab, storage_C_regulation):

        """
        Description
        ___________
        Nitrogen metabolism within local root segment cortex

        Parameters
        __________
        :param smax_struct : Maximal organic structure synthesis from mineral nitrogen and labil C (mol.s-1)
        :param affinity_Nm_struct :
        :param affinity_C_struct :

        """
        # No order in update propagation
        for vid in self.vertices:

            # if root segment emerged
            if self.struct_mass[vid] > 0:
                # amino acid synthesis
                self.AA_synthesis[vid] = smax_AA / (
                    ((1 + affinity_Nm_AA) / self.Nm[vid])
                    + ((1 + affinity_C_AA) / self.C_hexose_root[vid])
                )

                # Organic structure synthesis
                self.struct_synthesis[vid] = (smax_struct * self.AA[vid]
                                               / (affinity_AA_struct + self.AA[vid]))

                # Organic storage synthesis (Michaelis-Menten kinetic)
                self.storage_synthesis[vid] = (smax_stor * self.AA[vid]
                                               /(affinity_AA_stor + self.AA[vid]))

                # Organic storage catabolism
                Km_stor_root = affinity_stor_catab * np.exp(storage_C_regulation * self.C_hexose_root[vid])
                self.storage_catabolism[vid] = cmax_stor * self.C_hexose_reserve[vid] / (
                        Km_stor_root + self.C_hexose_reserve[vid])

                # AA catabolism
                Km_stor_root = affinity_AA_catab * np.exp(storage_C_regulation * self.C_hexose_root[vid])
                self.AA_catabolism[vid] = cmax_AA * self.AA[vid] / (
                        Km_stor_root + self.AA[vid])

    def update_N(self, r_Nm_AA, r_AA_struct, r_AA_stor, xylem_to_root, phloem_to_root, time_step):
        """
        Description
        ___________

        Parameters
        __________
        :param r_Nm_struct : Nm mol consumed per mol of organic structural synthesis (adim)

        """

        # No order in update propagation
        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:

                # Local nitrogen concentration update
                self.Nm[vid] += time_step / self.struct_mass[vid] * (
                        self.influx_Nm[vid]
                        + self.diffusion_Nm_phloem[vid]
                        - self.loading_Nm[vid]
                        - self.AA_synthesis[vid] * r_Nm_AA
                        + self.AA_catabolism[vid] / r_Nm_AA)

                self.AA[vid] += time_step / self.struct_mass[vid] * (
                        self.diffusion_AA_phloem[vid]
                        - self.diffusion_AA_soil[vid]
                        - self.loading_AA[vid]
                        + self.AA_synthesis[vid]
                        - self.struct_synthesis[vid] * r_AA_struct
                        - self.storage_synthesis[vid] * r_AA_stor
                        + self.storage_catabolism[vid] / r_AA_stor
                        - self.AA_catabolism[vid])

                # Global vessels' structural mass update
                self.xylem_struct_mass[vid] = self.struct_mass[vid] * xylem_to_root
                self.phloem_struct_mass[vid] = self.struct_mass[vid] * phloem_to_root

                # Global vessel's nitrogen pool update
                self.xylem_Nm[vid] += time_step / self.xylem_struct_mass[vid] * (
                        self.loading_Nm[vid]
                        + self.axial_diffusion_Nm_xylem[vid])
                self.xylem_AA[vid] += time_step / self.xylem_struct_mass[vid] * (
                        self.loading_AA[vid]
                        + self.axial_diffusion_AA_xylem[vid])
                self.phloem_Nm[vid] -= time_step / self.xylem_struct_mass[vid] * (
                        self.diffusion_Nm_phloem[vid]
                        + self.axial_diffusion_Nm_phloem[vid])
                self.phloem_AA[vid] -= time_step / self.xylem_struct_mass[vid] * (
                        self.diffusion_AA_phloem[vid]
                        + self.axial_diffusion_AA_phloem[vid])

        # Update plant-level properties
        self.xylem_Nm[1] -= time_step * self.Nm_root_shoot_xylem[1] / self.xylem_struct_mass[1]
        self.xylem_AA[1] -= time_step * self.AA_root_shoot_xylem[1] / self.xylem_struct_mass[1]
        self.phloem_Nm[1] += time_step * self.Nm_root_shoot_phloem[1] / self.phloem_struct_mass[1]
        self.phloem_AA[1] += time_step * self.AA_root_shoot_phloem[1] / self.phloem_struct_mass[1]

