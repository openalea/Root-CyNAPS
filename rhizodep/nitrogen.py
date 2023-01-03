"""
rhizodep.nitrogen
_________________
This is the main nitrogen cycle module for roots

Documentation and features
__________________________

Main functions
______________
Classes' names reprensent accounted hypothesis in the progressive development of the model.
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
import rhizodep.parameters_nitrogen as Nparam


class ContinuousVessels:

    def __init__(self, g, Nm, influx_Nm, loading_Nm, diffusion_Nm_phloem, struct_synthesis, storage_synthesis,
                 storage_catabolism, xylem_Nm, xylem_volume, phloem_Nm, phloem_volume, Nm_root_shoot_xylem,
                 Nm_root_shoot_phloem):

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
                        influx_Nm=influx_Nm,
                        loading_Nm=loading_Nm,
                        diffusion_Nm_phloem=diffusion_Nm_phloem,
                        struct_synthesis=struct_synthesis,
                        storage_synthesis=storage_synthesis,
                        storage_catabolism=storage_catabolism,
                        xylem_Nm=xylem_Nm,
                        xylem_volume=xylem_volume,
                        phloem_Nm=phloem_Nm,
                        phloem_volume=phloem_volume,
                        Nm_root_shoot_xylem=Nm_root_shoot_xylem,
                        Nm_root_shoot_phloem=Nm_root_shoot_phloem
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
                Nm
                volume
                influx_Nm
                loading_Nm
                diffusion_Nm_phloem
                struct_synthesis
                storage_synthesis
                storage_catabolism
                xylem_Nm
                xylem_volume
                phloem_Nm
                phloem_volume
                Nm_root_shoot_xylem
                Nm_root_shoot_phloem
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

    def transport_N(self, affinity_Nm_root, vmax_Nm_emergence, affinity_Nm_xylem, diffusion_phloem, transport_C_regulation,
                    transport_N_regulation, xylem_to_root, phloem_to_root, epiderm_differentiation, endoderm_differentiation):

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
                # We define nitrogen active uptake from soil
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

    def metabolism_N(self, smax_struct, affinity_Nm_struct, affinity_C_struct, cmax_stor, affinity_stor_catab, storage_C_regulation):

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
                # Organic structure synthesis
                self.struct_synthesis[vid] = smax_struct / (
                    ((1 + affinity_Nm_struct) / self.Nm[vid])
                    + ((1 + affinity_C_struct) / self.C_hexose_root[vid])
                )

                # Organic storage synthesis
                self.storage_synthesis[vid] = smax_struct / (
                        ((1 + affinity_Nm_struct) / self.Nm[vid])
                        + ((1 + affinity_C_struct) / self.C_hexose_root[vid])
                )

                # Organic storage catabolism
                Km_stor_root = affinity_stor_catab * np.exp(storage_C_regulation * self.C_hexose_root[vid])
                self.storage_catabolism[vid] = cmax_stor * self.C_hexose_reserve[vid] / (
                        Km_stor_root + self.C_hexose_reserve[vid])

    def update_N(self, r_Nm_struct, r_Nm_stor, xylem_to_root, phloem_to_root, time_step):
        """
        Description
        ___________

        Parameters
        __________
        :param r_Nm_struct : Nm mol consumed per mol of organic structural synthesis (adim)

        """
        # We define xylem and phloem nitrogen content (mol) from previous volume and concentrations.
        xylem_Nm_content = self.xylem_Nm[1] * self.xylem_volume[1]
        phloem_Nm_content = self.phloem_Nm[1] * self.phloem_volume[1]


        # Computing actualised volumes
        self.xylem_volume[1] = 0
        self.phloem_volume[1] = 0


        # No order in update propagation
        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:
                # Local volume update
                Nm_content = self.Nm[vid] * self.volume[vid]
                self.volume[vid] = (( np.pi * self.length[vid] * (self.radius[vid]) ** 2 )
                                   * (1 - (xylem_to_root ** 2) - (phloem_to_root ** 2)))

                # Local nitrogen concentration update
                self.Nm[vid] = Nm_content + (time_step / self.volume[vid]) * (
                        self.influx_Nm[vid]
                        + self.diffusion_Nm_phloem[vid]
                        - self.loading_Nm[vid]
                        - self.struct_synthesis[vid] * r_Nm_struct
                        - self.storage_synthesis[vid] * r_Nm_stor
                        + self.storage_catabolism[vid] / r_Nm_stor)

                # Global vessel's nitrogen pool update
                xylem_Nm_content += time_step * self.loading_Nm[vid]
                phloem_Nm_content -= time_step * self.diffusion_Nm_phloem[vid]
                self.xylem_volume[1] += np.pi * self.length[vid] * (self.radius[vid] * xylem_to_root) ** 2
                self.phloem_volume[1] += np.pi * self.length[vid] * (self.radius[vid] * phloem_to_root) ** 2

        # Update plant-level properties
        self.xylem_Nm[1] = (xylem_Nm_content - time_step * self.Nm_root_shoot_xylem[1]) / self.xylem_volume[1]
        self.phloem_Nm[1] = (phloem_Nm_content + time_step * self.Nm_root_shoot_phloem[1]) / self.phloem_volume[1]

