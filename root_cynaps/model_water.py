import numpy as np
from dataclasses import dataclass, field

from openalea.mtg.traversal import pre_order


# Properties' initialization

def set_value(value: float, min_value: float, max_value: float) -> float:
    if min_value <= value <= max_value:
        return value
    else:
        raise ValueError("The provided value is outside boundaries")


@dataclass
class RootWaterModel:
    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

    # FROM SOIL MODEL
    soil_water_pressure: float = field(default=0., metadata=dict(unit="Pa", unit_comment="of water", description="", value_comment="", references="", variable_type="input", by="model_soil"))
    soil_temperature: float = field(default=0., metadata=dict(unit="°C", unit_comment="", description="", value_comment="", references="", variable_type="input", by="model_soil"))

    # FROM ANATOMY MODEL
    xylem_volume: float = field(default=0., metadata=dict(unit="m3", unit_comment="", description="", value_comment="", references="", variable_type="input", by="model_anatomy"))
    cortex_exchange_surface: float = field(default=0., metadata=dict(unit="m2", unit_comment="", description="", value_comment="", references="", variable_type="input", by="model_anatomy"))
    apoplasmic_exchange_surface: float = field(default=0., metadata=dict(unit="m2", unit_comment="", description="", value_comment="", references="", variable_type="input", by="model_anatomy"))
    apoplasmic_stele: float = field(default=0., metadata=dict(unit="adim", unit_comment="", description="", value_comment="", references="", variable_type="input", by="model_anatomy"))

    # FROM GROWTH MODEL
    length: float = field(default=0, metadata=dict(unit="m", unit_comment="of root segment", description="", value_comment="", references="", variable_type="input", by="model_growth"))
    radius: float = field(default=0, metadata=dict(unit="m", unit_comment="of root segment", description="", value_comment="", references="", variable_type="input", by="model_growth"))
    struct_mass: float = field(default=0, metadata=dict(unit="g", unit_comment="of dry weight", description="", value_comment="", references="", variable_type="input", by="model_growth"))

    # FROM SHOOT MODEL
    water_root_shoot_xylem: float = field(default=0., metadata=dict(unit="mol.time_step-1", unit_comment="of water", description="", value_comment="", references="", variable_type="input", by="model_shoot"))

    # --- INITIALIZE MODEL STATE VARIABLES ---

    # LOCAL VARIABLES

    # Pools initial values
    xylem_water: float = field(default=set_value(0., min_value=0., max_value=1e-6), metadata=dict(unit="mol", unit_comment="of water", description="", value_comment="", references="", variable_type="state_variable", by="model_water"))

    # Water transport processes
    radial_import_water: float = field(default=0., metadata=dict(unit="mol.time_step-1", unit_comment="of water", description="", value_comment="", references="", variable_type="state_variable", by="model_water"))
    shoot_uptake: float = field(default=0., metadata=dict(unit="mol", unit_comment="of water", description="", value_comment="", references="", variable_type="state_variable", by="model_water"))
    axial_export_water_up: float = field(default=0., metadata=dict(unit="mol", unit_comment="of water", description="", value_comment="", references="", variable_type="state_variable", by="model_water"))
    axial_import_water_down: float = field(default=0., metadata=dict(unit="mol", unit_comment="of water", description="", value_comment="", references="", variable_type="state_variable", by="model_water"))

    # SUMMED STATE VARIABLES

    xylem_total_water: float = field(default=0., metadata=dict(unit="mol", unit_comment="of water", description="", value_comment="", references="", variable_type="plant_scale_state", by="model_water"))
    xylem_total_pressure: float = field(default=set_value(-0.1e6, min_value=-0.5e6, max_value=-0.05e6), metadata=dict(unit="Pa", unit_comment="", description="apoplastic pressure in stele at rest, we want the -0.5e6 target to be emerging from water balance", value_comment="", references="", variable_type="plant_scale_state", by="model_water"))

    # --- INITIALIZES MODEL PARAMETERS ---

    # time resolution
    sub_time_step: int = field(default=set_value(3600, min_value=1, max_value=24 * 3600), metadata=dict(unit="s", unit_comment="", description="MUST be a multiple of base time_step", value_comment="", references="", variable_type="parameter", by="model_water"))

    # Water properties
    water_molar_mass: float = field(default=18, metadata=dict(unit="g.mol-1", unit_comment="", description="", value_comment="", references="", variable_type="parameter", by="model_water"))
    water_volumic_mass: float = field(default=1e6, metadata=dict(unit="g.m-3", unit_comment="", description="", value_comment="", references="", variable_type="parameter", by="model_water"))
    sap_viscosity: float = field(default=1.3e6, metadata=dict(unit="Pa", unit_comment="", description="", value_comment="", references="", variable_type="parameter", by="model_water"))

    # Vessel mechanical properties
    xylem_young_modulus: float = field(default=1e6, metadata=dict(unit="Pa", unit_comment="", description="radial elastic modulus of xylem tissues (Has to be superior to initial difference between root and soil)", value_comment="", references="", variable_type="parameter", by="model_water"))
    xylem_cross_area_ratio: float = field(default=10, metadata=dict(unit="adim", unit_comment="", description="0.84 * (0.36 ** 2) apoplasmic cross-section area ratio * stele radius ratio^2 # TODO : rename buffer ratio", value_comment="", references="", variable_type="parameter", by="model_water"))

    cortex_water_conductivity: float = field(default=1e-14 * 1e5, metadata=dict(unit="m.s-1.Pa-1", unit_comment="", description="", value_comment="", references="", variable_type="parameter", by="model_water"))
    apoplasmic_water_conductivity: float = field(default=1e-14 * 1e6, metadata=dict(unit="m.s-1.Pa-1", unit_comment="", description="", value_comment="", references="", variable_type="parameter", by="model_water"))
    xylem_tear: float = field(default=9e5, metadata=dict(unit="Pa", unit_comment="", description="maximal difference with soil pressure before xylem tearing (absolute, < xylem_young modulus)", value_comment="", references="", variable_type="parameter", by="model_water"))

    def __init__(self, g, time_step, sub_time_step):
        """
        Description :

        This root water model discretized at root segment's scale intends to account for heterogeneous axial and radial water flows observed in the roots (Bauget et al. 2022).

        Parameters

        Hypothesis :

        Accounting for heterogeneous water flows would improbe the overall nutrient balance for root hydromineral uptake.
        """

        self.g = g
        self.props = self.g.properties()
        self.time_step = time_step
        self.sub_time_step = sub_time_step
        self.vertices = self.g.vertices(scale=self.g.max_scale())

        self.state_variables = [name for name, value in self.__dataclass__fields__ if value.metadata["variable_type"] == "state_variable"]
        print(self.state_variables)

        for name in self.state_variables:
            if name not in self.props.keys():
                self.props.setdefault(name, {})
            # set default in mtg
            self.props[name].update({key: getattr(self, name) for key in self.vertices})
            # link mtg dict to self dict
            setattr(self, name, self.props[name])

        # Repeat the same process for total root system properties
        self.plant_scale_states = [name for name, value in self.__dataclass__fields__ if value.metadata["variable_type"] == "plant_scale_state"]

        for name in self.plant_scale_states:
            if name not in self.props.keys():
                self.props.setdefault(name, {})
            # set default in mtg
            self.props[name].update({1: getattr(self, name)})
            # link mtg dict to self dict
            setattr(self, name, self.props[name])

        # Select real children for collar element (vid == 1).
        # This is mandatory for correct collar-to-tip Hagen-Poiseuille flow partitioning.
        self.collar_children, self.collar_skip = [], []
        for vid in self.vertices:
            child = self.g.children(vid)
            if (self.struct_mass[vid] == 0) and (True in [self.struct_mass[k] > 0 for k in child]):
                self.collar_skip += [vid]
                self.collar_children += [k for k in self.g.children(vid) if self.struct_mass[k] > 0]

    def init_xylem_water(self):
        # At pressure = soil_pressure, the corresponding xylem volume at rest is
        # filled with water in standard conditions

        # We compute the total water amount from the formula used for pressure calculation
        self.xylem_total_water[1] = ((((self.xylem_total_pressure[1] - np.mean(list(self.soil_water_pressure.values()))) / self.xylem_young_modulus) + 1) ** 2) * (
                np.pi * (np.mean(list(self.radius.values())) ** 2) * sum(self.length.values()) * self.xylem_cross_area_ratio * self.water_volumic_mass) / self.water_molar_mass

        sum_volume = sum(self.xylem_volume.values())

        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:
                self.xylem_water[vid] = self.xylem_total_water[1] * self.xylem_volume[vid] / sum_volume

    def transport_water(self):
        # Using previous time-step flows, we compute current time-step pressure for flows computation

        # Compute the minimal water content for current dimensions
        tearing_xylem_total_water = (((- self.xylem_tear / self.xylem_young_modulus) + 1) ** 2) * (
                  np.pi * (np.mean(list(self.radius.values())) ** 2) * sum(
              self.length.values()) * self.xylem_cross_area_ratio * self.water_volumic_mass) / self.water_molar_mass

        # we set collar element the flow provided by shoot model
        potential_transpiration = self.water_root_shoot_xylem[1] * self.sub_time_step
        # condition if potential transpiration is going to lead to a tearing pressure of xylem
        if self.xylem_total_water[1] - potential_transpiration < tearing_xylem_total_water:
            actual_transpiration = self.xylem_total_water[1] - tearing_xylem_total_water
        else:
            actual_transpiration = potential_transpiration

        self.axial_export_water_up[1] = actual_transpiration

        # Loop computing individual segments' water exchange
        for vid in self.vertices:
            # radial exchanges are only hydrostatic-driven for now
            apoplastic_water_import = self.apoplasmic_water_conductivity * (self.soil_water_pressure[vid] - self.xylem_total_pressure[1]) * self.apoplasmic_exchange_surface[vid]

            cross_membrane_water_import = self.cortex_water_conductivity * (self.soil_water_pressure[vid] - self.xylem_total_pressure[1]) * self.cortex_exchange_surface[vid]

            self.radial_import_water[vid] = (apoplastic_water_import + cross_membrane_water_import) * self.sub_time_step
            # We suppose uptake is evenly reparted over the xylem to avoid over contribution of apexes in
            # the down propagation of transpiration (computed below)
            self.shoot_uptake[vid] = self.axial_export_water_up[1] * self.xylem_water[vid] / self.xylem_total_water[1]

        # Finally we compute the axial result of these transpiration fluxes and radial uptake
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

                # If this is a root tip or a non-emerged root segment, there is no down import
                if (vid != 1) and ((len(child) == 0) or (True not in [self.struct_mass[k] > 0 for k in child])):
                    self.axial_import_water_down[vid] = 0

                # if there are children, there is a down import flux
                else:
                    self.axial_import_water_down[vid] = self.axial_export_water_up[vid] - self.shoot_uptake[vid]

                # water balance is computed here to prevent another for loop over mtg
                self.xylem_water[vid] += self.radial_import_water[vid] - self.shoot_uptake[vid]
                # For reference, balance between flows is :
                # self.xylem_water[vid] = self.xylem_water[vid] + self.radial_import_water[vid] - self.shoot_uptake[vid] = \
                #   = self.xylem_water[vid] + self.radial_import_water[vid] + self.axial_import_water_down[vid] - self.axial_export_water_up[vid]

                # For current vertex's children, provide previous down flow as axial upper flow for children
                # if current vertex is collar, we affect down flow at previously computed collar children
                if vid == 1:
                    HP = [0 for k in self.collar_children]
                    for k in range(len(self.collar_children)):
                        # compute Hagen-Poiseuille coefficient
                        HP[k] = np.pi * (self.radius[self.collar_children[k]]**4) / (8 * self.sap_viscosity)
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
                        # compute Hagen-Poiseuille coefficient
                        HP[k] = np.pi * (self.radius[child[k]]**4) / (8 * self.sap_viscosity)

                    HP_tot = sum(HP)
                    for k in range(len(child)):
                        self.axial_export_water_up[child[k]] = (HP[k] / HP_tot) * self.axial_import_water_down[vid]

        self.update_sums()

        # Finally, we assume pressure homogeneity and compute the resulting pressure for the next time_step
        self.xylem_total_pressure[1] = self.xylem_young_modulus * (
                    (((self.xylem_total_water[1] * self.water_molar_mass) / (
                            np.pi * (np.mean(list(self.radius.values())) ** 2) * sum(self.length.values()) *
                            self.xylem_cross_area_ratio * self.water_volumic_mass)) ** 0.5) - 1) + np.mean(
                            list(self.soil_water_pressure.values()))

    def update_sums(self):
        self.xylem_total_water[1] = sum(self.xylem_water.values())

    def add_properties_to_new_segments(self):
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        for vid in self.vertices:
            if vid not in list(self.xylem_water.keys()):
                for prop in list(self.keywords.keys()):
                    getattr(self, prop)[vid] = 0

    def exchanges_and_balance(self):
        """
        Description
        ___________
        Model processes and balance for water to be called by simulation files.

        """
        for k in range(int(self.time_step/self.sub_time_step)):
            self.add_properties_to_new_segments()
            self.transport_water()
            self.update_sums()

