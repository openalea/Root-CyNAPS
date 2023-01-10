"""
rhizodep.water
_________________
This is the main water cycle module for roots

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


class DiscreteWaterVessels:

    def __init__(self, g, W, W_xylem, W_potential, W_potential_xylem, pressure, pressure_xylem, influx_W, loading_W):

        """
        Description
        Initialization of nitrogen-related variables

        Parameters
        :param g: MTG

        Hypothesis
        H1 :
        H2 :
        """

        self.g = g
        # New properties' creation in MTG
        keywords = dict(W=W,
                        W_xylem=W_xylem,
                        W_potential=W_potential,
                        pressure=pressure,
                        pressure_xylem=pressure_xylem,
                        W_potential_xylem=W_potential_xylem,
                        influx_W=influx_W,
                        loading_W=loading_W
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
                        soil_W
                        W
                        W_xylem
                        W_potential
                        W_potential_xylem
                        pressure
                        pressure_xylem
                        influx_W
                        loading_W
                        length
                        radius
                        struct_mass
                        """.split()

                for name in states:
                    setattr(self, name, props[name])

                # Note : Global properties are declared as local ones, but only vertice 1 will be updated

    def transport_W(self, k, xylem_to_root):

        # No order in update propagation
        for vid in self.vertices:

            # if root segment emerged
            if self.struct_mass[vid] > 0:

                # RADIAL TRANSPORT

                self.influx_W[vid] = (k * (self.soil_W[vid] - self.W_potential[vid])
                                     * (2 * np.pi * self.radius[vid] * self.length[vid]))

                self.loading_W[vid] = (k * (self.W_potential[vid] - self.W_potential_xylem[vid])
                                     * (2 * np.pi * self.radius[vid] * xylem_to_root * self.length[vid]))

    def update_W(self, R, MH2O, time_step):

        W_tot_xylem = 0
        V_tot_xylem = 0
        temp = 0

        # No order in update propagation
        for vid in self.vertices:

            # if root segment emerged
            if self.struct_mass[vid] > 0:

                self.W_potential[vid] = (self.pressure[vid] + R * self.temperature[vid]
                                        * (self.Nm[vid] + self.C_hexose_root[vid]))

                self.W_potential_xylem[vid] = (self.pressure_xylem[1] + R * self.temperature[vid]
                                               * (self.xylem_Nm[vid]))

                self.W[vid] += time_step * (self.influx_W[vid] - self.loading_W[vid])

                self.W_xylem[vid] += time_step * self.loading_W[vid]

                self.volume[vid] = self.W[vid] * MH2O

                self.volume_xylem[vid] = self.W_xylem[vid] * MH2O

                self.pressure[vid] = self.W[vid] * R * self.temperature[vid] / self.volume[vid]

                W_tot_xylem += self.W_xylem[vid]
                V_tot_xylem += self.volume_xylem[vid]
                temp += self.temperature[vid]

        self.pressure_xylem[1] = W_tot_xylem * R * (temp / len(self.vertices)) / V_tot_xylem

