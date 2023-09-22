import numpy as np
from dataclasses import dataclass, asdict

from openalea.mtg.traversal import pre_order


# Properties' initialization


@dataclass
class InitWater:
    # time resolution
    sub_time_step: int = 3600 # (second) MUST be a multiple of base time_step
    # Pools
    xylem_water: float = 0  # (mol) water content
    water_molar_mass: float = 18    # g.mol-1
    water_volumic_mass: float = 1e6  # g.m-3
    xylem_total_pressure: float = -0.5e6  # (Pa) apoplastic pressure in stele
    # Water transports
    radial_import_water: float = 0
    axial_export_water_up: float = 0
    axial_import_water_down: float = 0
    # Mechanical properties
    xylem_young_modulus: float = 1e5  # (Pa) radial elastic modulus of xylem tissues
    xylem_cross_area_ratio: float = 1  # (adim) apoplasmic cross-section area ratio * stele radius ratio^2 # TODO : rename buffer ratio

@dataclass
class TransportWater:
    water_molar_mass: float = 18  # g.mol-1
    radial_water_conductivity: float = 1e-14 * 1e3  # m.s-1.Pa-1
    reflexion_coef: float = 0.85    # adim
    R: float = 8.314
    sap_viscosity: float = 1.3e6    # Pa


class WaterModel:
    def __init__(self, g, time_step, sub_time_step, xylem_water, water_molar_mass, water_volumic_mass, xylem_total_pressure,
                 radial_import_water, axial_export_water_up, axial_import_water_down, xylem_young_modulus, xylem_cross_area_ratio):
        """
                Description :
    	        
                This root water model discretized at root segment's scale intends to account for heterogeneous axial and radial water flows observed in the roots (Bauget et al. 2022).

                Parameters

                Hypothesis :
                
                Accounting for heterogeneous water flows would improbe the overall nutrient balance for root hydromineral uptake.
                """

        self.g = g
        self.time_step = time_step
        self.sub_time_step = sub_time_step
        self.xylem_young_modulus = xylem_young_modulus
        self.xylem_cross_area_ratio = xylem_cross_area_ratio

        # New spatialized properties' creation in MTG
        self.keywords = dict(
            xylem_water=xylem_water,
            radial_import_water=radial_import_water,
            axial_export_water_up=axial_export_water_up,
            axial_import_water_down=axial_import_water_down)

        props = self.g.properties()
        for name in self.keywords:
            props.setdefault(name, {})

        # vertices storage for future calls in for loops
        self.vertices = self.g.vertices(scale=g.max_scale())
        for vid in self.vertices:
            for name, value in self.keywords.items():
                # Effectively creates the new property
                props[name][vid] = value

        # Accessing properties once, pointing to g for further modifications
        self.states = """
                        C_hexose_soil
                        xylem_water
                        C_sucrose_root
                        radial_import_water
                        axial_export_water_up
                        axial_import_water_down
                        length
                        radius
                        struct_mass
                        living_root_hairs_external_surface
                        xylem_volume
                        """.split()

        # Declare MTG properties in self
        for name in self.states:
            setattr(self, name, props[name])

        # Repeat the same process for total root system properties

        # Creating variables for global balance and outputs
        self.totals_keywords = dict(xylem_total_water=0.001,
                                    xylem_total_volume=0,
                                    xylem_total_pressure=xylem_total_pressure)

        for name, value in self.totals_keywords.items():
            props.setdefault(name, {})
            props[name][1] = value

        # Accessing properties once, pointing to g for further modifications
        self.totals_states = """
                                xylem_total_water
                                xylem_total_volume
                                xylem_total_pressure
                                """.split()

        # Declare MTG properties in self
        for name in self.totals_states:
            setattr(self, name, props[name])

        # proper initialization of the xylem water content
        self.water_volumic_mass = water_volumic_mass

        # Declare to outside modules which variables are needed
        # TODO : convert to dict of dict for the builder to print variable expertise informations
        self.inputs = {
            "soil": [
                "soil_water_pressure",
                "soil_temperature"
            ],
            "structure": [
                "xylem_volume",
                "cylinder_exchange_surface",
                "apoplasmic_stele"
            ],
            "shoot_water": [
                "water_root_shoot_xylem"
            ]
        }

        # Select real children for collar element (vid == 1).
        # This is mandatory for right collar-to-tip Hagen-Poiseuille flow partitioning.
        self.collar_children, self.collar_skip = [], []
        for vid in self.vertices:
            child = self.g.children(vid)
            if (self.struct_mass[vid] == 0) and (True in [self.struct_mass[k] > 0 for k in child]):
                self.collar_skip += [vid]
                self.collar_children += [k for k in self.g.children(vid) if self.struct_mass[k] > 0]

    def init_xylem_water(self, water_molar_mass=18):
        # At pressure = soil_pressure, the corresponding xylem volume at rest is
        # filled with water in standard conditions
        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:
                self.xylem_water[vid] = ((((self.xylem_total_pressure[1] - self.soil_water_pressure[vid]) / self.xylem_young_modulus + 1)) ** 2) * (
                        np.pi * self.length[vid] * (self.radius[vid]**2) * self.xylem_cross_area_ratio *
                        self.water_volumic_mass) / water_molar_mass

        self.update_sums()

    def transport_water(self, water_molar_mass, radial_water_conductivity, reflexion_coef, R, sap_viscosity):
        # Using previous time-step flows, we compute current time-step pressure for flows computation

        # we set collar element the flow provided by shoot model
        potential_transpiration = self.water_root_shoot_xylem[1] * self.sub_time_step

        # radial exchanges are only hydrostatic-driven for now
        for vid in self.vertices:
            self.radial_import_water[vid] = radial_water_conductivity * (self.soil_water_pressure[vid] - self.xylem_total_pressure[1]) * self.cylinder_exchange_surface[vid] * self.sub_time_step
            self.xylem_water[vid] += self.radial_import_water[vid]

        # First we limit collar transpiration if the result exceeds xylem shear strength
        potential_pressure = self.xylem_young_modulus * (((((self.xylem_total_water - potential_transpiration + sum(self.radial_import_water.values())) * water_molar_mass) / (
                np.pi * (np.mean(list(self.radius.values()))**2) * sum(self.length.values()) * self.xylem_cross_area_ratio * self.water_volumic_mass))**0.5) - 1) + np.mean(list(self.soil_water_pressure.values()))

        shear_max = 1e6

        actual_pressure = []
        for vid in self.vertices:
            if abs(self.soil_water_pressure[vid] - potential_pressure) > shear_max:
                actual_pressure += [self.soil_water_pressure[vid] - shear_max]
                print(True)

        if len(actual_pressure) > 0:
            self.xylem_total_pressure[1] = min(actual_pressure)
            self.axial_export_water_up[1] = self.xylem_total_water + sum(self.radial_import_water.values()) - ((((actual_pressure - np.mean(list(self.soil_water_pressure.values()))) / self.xylem_young_modulus + 1)**2) * (
                (np.pi * (np.mean(list(self.radius.values())) ** 2) * sum(self.length.values()) * self.xylem_cross_area_ratio * self.water_volumic_mass) / water_molar_mass))
            
        else:
            self.xylem_total_pressure[1] = -0.5e6 # potential_pressure
            self.axial_export_water_up[1] = potential_transpiration

        self.xylem_total_water += sum(self.radial_import_water.values()) - self.axial_export_water_up[1]
        #print(sum(self.radial_import_water.values()), self.axial_export_water_up[1])

        # Finally we compute the axial result of these transpiration and radial uptake
        # We define "root" as the starting point of the loop below:
        root_gen = self.g.component_roots_at_scale_iter(self.g.root, scale=1)
        root = next(root_gen)

        # We travel in the MTG from the root collar to the tips:
        for vid in pre_order(self.g, root):
            # We apply the following for all structural mass, because null length element can be support for
            # ramification

            child = self.g.children(vid)

            # if we look at a collar artificial vertex of null lenght, we do nothing
            if vid not in self.collar_skip:

                # For current vertex, compute axial down flow from axial upper flow, radial flow and volume at considered pressure
                # There is no pressure variation effect as water is incompressible
                # if this is a root tip, there is no down import flux

                # Assuming pressure homogeneity, we compute segment's final water content from Young's elastic modulus theory

                final_xylem_water = ((((self.xylem_total_pressure[1] - self.soil_water_pressure[vid]) / self.xylem_young_modulus + 1)) ** 2) * (
                        np.pi * self.length[vid] * (self.radius[vid]**2) * self.xylem_cross_area_ratio *
                        self.water_volumic_mass) / water_molar_mass

                # If this is a root tip or a non-emerged root segment, there is no down import flux.
                if (vid != 1) and ((len(child) == 0) or (True not in [self.struct_mass[k] > 0 for k in child])):
                    self.axial_import_water_down[vid] = 0

                # if there are children, there is a down import flux
                else:
                    self.axial_import_water_down[vid] = final_xylem_water - self.xylem_water[vid] + self.axial_export_water_up[vid]
                    # water balance is computed here to prevent another for loop over mtg
                self.xylem_water[vid] = final_xylem_water

                #if vid == 193 :
                #    print(self.xylem_water[vid], self.axial_export_water_up[vid], self.radial_import_water[vid], self.axial_import_water_down[vid])

                # For current vertex's children, provide previous down flow as axial upper flow for children
                # if current vertex is collar, we affect down flow at previously computed collar children
                if vid == 1:
                    HP = [0 for k in self.collar_children]
                    for k in range(len(self.collar_children)):
                        # compute Hagen-Poiseuille coefficient
                        HP[k] = np.pi * (self.radius[self.collar_children[k]]**4) / (8 * sap_viscosity)
                    HP_tot = sum(HP)
                    for k in range(len(self.collar_children)):
                        self.axial_export_water_up[self.collar_children[k]] = (HP[k] / HP_tot) * self.axial_import_water_down[vid]

                # else, if there is only one child, the entire flux is applied
                elif len(child) == 1:
                    self.axial_export_water_up[child[0]] = self.axial_import_water_down[vid]

                # finally, if there are several children, a fraction of the flux is applied
                # according to Hagen-Poiseuille's law (same as collar)
                else:
                    HP = [0 for k in child]
                    for k in range(len(child)):
                        if self.struct_mass[child[k]] > 0:
                            # compute Hagen-Poiseuille coefficient
                            HP[k] = np.pi * (self.radius[child[k]]**4) / (8 * sap_viscosity)

                    HP_tot = sum(HP)
                    for k in range(len(child)):
                        self.axial_export_water_up[child[k]] = (HP[k] / HP_tot) * self.axial_import_water_down[vid]

    def update_sums(self):
        self.xylem_total_water = sum(self.xylem_water.values())
        self.xylem_total_volume = sum(self.xylem_volume.values())

    def exchanges_and_balance(self):
        """
        Description
        ___________
        Model processes and balance for water to be called by simulation files.

        """
        for k in range(int(self.time_step/self.sub_time_step)):
            self.transport_water(**asdict(TransportWater()))
            self.update_sums()

