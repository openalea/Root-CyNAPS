import numpy as np
from dataclasses import dataclass, asdict

from openalea.mtg.traversal import pre_order


# Properties' initialization


@dataclass
class InitWater:
    # Pools
    xylem_water: float = 0
    # xylem_pressure : float = 101325 # (Pa) apoplastic pressure in stele

@dataclass
class TransportWater:
    radial_water_conductivity: float = 0


class WaterModel:
    def __int__(self, g, time_step, xylem_water, radial_import_water, axial_export_water_up,
                axial_import_water_down, water_root_shoot_xylem):
        """
                Description

                Parameters

                Hypothesis
                """

        self.g = g
        self.time_step = time_step

        # New properties' creation in MTG
        self.keywords.update(dict(
            xylem_water=xylem_water,
            radial_import_water=radial_import_water,
            axial_export_water_up=axial_export_water_up,
            axial_import_water_down=axial_import_water_down))

        # Creating variables for
        self.root_system_totals.update(dict(xylem_total_water=0,
                                            xylem_total_volume=0,
                                            xylem_total_pressure=101325
                                            ))

        self.shoot_exchanges = dict(water_root_shoot_xylem=water_root_shoot_xylem)

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
        self.states += """
                                soil_water_pressure
                                soil_temperature
                                C_hexose_soil
                                xylem_water
                                xylem_volume
                                C_sucrose_root
                                radial_import_water
                                axial_export_water_up
                                axial_import_water_down
                                root_exchange_surface
                                stele_exchange_surface
                                apoplasmic_stele
                                length
                                radius
                                struct_mass
                                """.split()

        # Declare MTG properties in self
        for name in self.states:
            setattr(self, name, props[name])

        # Declare exchanges with flow retreived from the shoot model
        for name in self.shoot_exchanges:
            setattr(self, name, self.shoot_exchanges[name])

        # Declare totals computed for global model's outputs
        for name in self.root_system_totals:
            setattr(self, name, self.root_system_totals[name])

        # proper initialization of the xylem water content
        self.init_xylem_water(R=8.314)

    def init_xylem_water(self, R):
        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:
                self.xylem_water[vid] = self.xylem_volume[vid] * self.xylem_total_pressure / (R * self.soil_temperature[vid])

    def transport_water(self, radial_water_conductivity, reflexion_coef, R, sap_viscosity):
        # Spatialized for all root segments in MTG...
        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:
                # Here radial flow if derived from hydraulic potential differencies over the time step
                self.radial_import_water[vid] = self.time_step * radial_water_conductivity * (
                        self.soil_water_pressure[vid] - self.xylem_total_pressure) * reflexion_coef * R * \
                                                self.soil_temperature[vid] * (
                                                        self.soil_hexose[vid] - self.C_sucrose_root[vid]) * \
                                                self.stele_exchange_surface[vid]

        # First balance to compute axial transport
        next_xylem_total_pressure = self.xylem_total_water - self.water_root_shoot_xylem * self.time_step + sum(
            self.radial_import_water.values()) * (
                                            R * self.soil_temperature[0]) / self.xylem_total_volume

        # We define "root" as the starting point of the loop below:
        root_gen = self.g.component_roots_at_scale_iter(self.g.root, scale=1)
        root = next(root_gen)

        self.axial_export_water_up[1] = self.water_root_shoot_xylem * self.time_step
        # We travel in the MTG from the root collar to the tips:
        for vid in pre_order(self.g, root):
            # if root segment emerged
            if self.struct_mass[vid] > 0:

                child = self.g.children(vid)
                # if this is a root tip, there is no down import flux
                if len(child) == 0:
                    self.axial_import_water_down[vid] = 0
                # if there are children who actually emerged, there is a down import flux
                elif 0 not in [self.struct_mass[k] for k in child]:
                    self.axial_import_water_down[vid] = self.axial_export_water_up[vid] - self.radial_import_water[
                        vid] + (
                                                                (
                                                                            next_xylem_total_pressure - self.xylem_total_pressure) *
                                                                self.xylem_volume[vid] / (
                                                                        R * self.soil_temperature[vid]))
                # if there are children who did not emerge, there is no down import flux
                else:
                    self.axial_import_water_down[vid] = 0

                # if there is only one child, the entire flux is applied
                if len(child) == 1:
                    self.axial_export_water_up[child[0]] = self.axial_import_water_down[vid]

                # if there are several children, a fraction of the flux is applied according to Hagen-Poiseuille's law
                else:
                    HP = [0 for k in child]
                    for k in range(len(child)):
                        if self.struct_mass[k] > 0:
                            # compute Hagen-Poiseuille coefficient
                            HP[k] = np.pi * self.radius[child[k]] / (8 * sap_viscosity)
                    HP_tot = sum(HP)
                    for k in range(len(child)):
                        self.axial_export_water_up[child[k]] = (HP[k] / HP_tot) * self.axial_import_water_down[vid]

        self.xylem_total_pressure = next_xylem_total_pressure

    def Update_water_local(self, v):
        self.xylem_water[v] += (self.radial_import_water[v]
                                - self.axial_export_water_up[v]
                                + self.axial_import_water_down[v]
                                )

    def update_sums(self):
        self.xylem_total_water = sum(self.xylem_water.values())
        self.xylem_total_volume = sum(self.xylem_volume.values())

    def exchanges_and_balance(self):
        """
        Description
        ___________
        Model processes and balance for water to be called by simulation files.

        """
        self.transport_water(**asdict(TransportWater()))
        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:
                self.Update_water_local(vid)

        self.update_sums()
