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

import pickle
import numpy as np



class continuous_vessels:

    def __init__(self,
                 g,
                 soil_Nm: float = 1,
                 Nm: float = 1e-5,
                 xylem_Nm: float = 0.1,
                 xylem_volume: float = 5e-10,
                 influx_Nm: float = 0,
                 loading_Nm: float = 0):

        """
        Description
        Initialization of nitrogen-related variables

        Parameters
        :param g: MTG
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
        self.g = g
        # New properties' creation in MTG
        keywords = dict(soil_Nm=soil_Nm,
                    Nm=Nm,
                    influx_Nm=influx_Nm,
                    loading_Nm=loading_Nm)

        props = self.g.properties()
        for name in keywords:
            props.setdefault(name, {})

        self.vertices = self.g.vertices(scale=g.max_scale())
        for vid in self.vertices:
            for name, value in keywords.items():
                props[name][vid] = value

        # Accessing properties once, pointing to g
        # N related
        # main model related
        states = """
                soil_Nm
                Nm
                influx_Nm
                loading_Nm
                length
                radius
                struct_mass
                C_hexose_root
                thermal_time_since_emergence
                """.split()
        for name in states:
            setattr(self, name, props[name])

        # global vessel's property creation in first node
        props['xylem_Nm'] = {0: xylem_Nm}
        props['xylem_volume'] = {0: xylem_volume}

        # Accessing global pools once
        self.xylem_Nm = props['xylem_Nm'][0]
        self.xylem_volume = props['xylem_volume'][0]



    def transport_N(self,
                    # kinetic parameters
                    affinity_Nm_root: float = 1,
                    vmax_Nm_emergence: float = 0.1,
                    affinity_Nm_xylem: float = 0.1,
                    # metabolism-related parameters
                    transport_C_regulation: float = 1e-2,
                    transport_N_regulation:float = 0.01,
                    # architecture parameters
                    xylem_to_root: float = 0.2,
                    epiderm_differentiation: float = 1e-6,
                    endoderm_differentiation: float = 1e-6
                    ):
        """
        Description
        ___________
        Nitrogen transport between local soil, local root segment and global vessels (xylem and phloem).

        Parameters
        __________
        :param g: MTG
        :param affinity_Nm_root: Active transport from soil Km parameter (mol.m-3)
        :param vmax_Nm_emergence: Surfacic maximal active transport rate in roots (mol.m-2.s-1)
        :param affinity_Nm_xylem: Active transport from root Km parameter (mol.g-1)
        :param transport_C_regulation: Affinity coefficient for the nitrogen active transport regulation function by root C (mol.g-1) (?)
        :param xylem_to_root: Radius ratio between mean xylem and root segment (adim)
        :param epiderm_differentiation: Epiderm differentiation rate (°C-1.d-1)
        :param endoderm_differentiation: Endoderm differentiation rate (°C-1.d-1)

        Hypothesis
        __________
        H1: We summarize radial active transport controls (transporter density, affinity regulated with genetics
        or environnemental control, etc) as one mean transporter following Michaelis Menten's model.

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
                vmax_Nm_root = vmax_Nm_emergence * np.exp(- epiderm_differentiation * self.thermal_time_since_emergence[vid])
                # Km is supposed affected by different processes regulated by destination nitrogen availability
                # (HATS/LATS composition and availability, phosphorylation, etc)
                km_Nm_root = affinity_Nm_root * np.exp(transport_N_regulation * self.Nm[vid])
                # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
                self.influx_Nm[vid] = ((self.soil_Nm[vid] * vmax_Nm_root / (self.soil_Nm[vid] + km_Nm_root))
                                 * (2 * np.pi * self.radius[vid] * self.length[vid])
                                 * (self.C_hexose_root[vid] / (self.C_hexose_root[vid] + transport_C_regulation)))

                # We define active xylem loading from root segment
                # Vmax supposed affected by root aging
                vmax_Nm_xylem = vmax_Nm_emergence * np.exp(- endoderm_differentiation * self.thermal_time_since_emergence[vid])
                # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
                self.loading_Nm[vid] = ((self.Nm[vid] * vmax_Nm_xylem / (self.Nm[vid] + affinity_Nm_xylem))
                                  * (2 * np.pi * self.radius[vid] * xylem_to_root * self.length[vid])
                                  * (self.C_hexose_root[vid] / (self.C_hexose_root[vid] + transport_C_regulation)))

                # print(influx_N[vid], loading_N[vid])


    def metabolism_N(self):
        return 1


    def update_N(self,
                 xylem_to_root=0.2,
                 time_step=3600):

        # We define xylem nitrogen content (mol) from previous volume and concentrations.
        xylem_Nm_content = self.xylem_Nm * self.xylem_volume
        # Computing actualised volume
        self.xylem_volume = 0

        # No order in update propagation
        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:
                # Local nitrogen pool update
                self.Nm[vid] += (time_step / self.struct_mass[vid]) * (self.influx_Nm[vid] - self.loading_Nm[vid])

                # Global vessel's nitrogen pool update
                xylem_Nm_content += time_step * self.loading_Nm[vid]
                self.xylem_volume += np.pi * self.length[vid] * (self.radius[vid] * xylem_to_root) ** 2

        # Update plant-level properties
        self.xylem_Nm = xylem_Nm_content / self.xylem_volume

        return self.g