import numpy as np
from openalea.mtg.traversal import pre_order2
from dataclasses import dataclass

from metafspm.component import Model, declare
from metafspm.component_factory import *


family = "hydraulic"


@dataclass
class RootWaterModel(Model):

    family = "hydraulic"

    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

    # FROM SOIL MODEL
    soil_water_pressure: float = declare(default=0., unit="Pa", unit_comment="of water", description="", 
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="input", by="model_soil", state_variable_type="", edit_by="user")
    soil_temperature: float = declare(default=0., unit="°C", unit_comment="", description="", 
                                      min_value="", max_value="", value_comment="", references="", DOI="",
                                      variable_type="input", by="model_soil", state_variable_type="", edit_by="user")
    soil_solutes: float = declare(default=0., unit="mol.m-3", unit_comment="", description="Total soil solute volumic concentration", 
                                      min_value="", max_value="", value_comment="", references="", DOI="",
                                      variable_type="input", by="model_soil", state_variable_type="", edit_by="user")

    # FROM ANATOMY MODEL
    xylem_volume: float = declare(default=0., unit="m3", unit_comment="", description="", 
                                  min_value="", max_value="", value_comment="", references="", DOI="",
                                  variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    cortex_exchange_surface: float = declare(default=0., unit="m2", unit_comment="", description="", 
                                             min_value="", max_value="", value_comment="", references="", DOI="",
                                             variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    apoplasmic_exchange_surface: float = declare(default=0., unit="m2", unit_comment="", description="", 
                                                 min_value="", max_value="", value_comment="", references="", DOI="",
                                                 variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")

    # FROM GROWTH MODEL
    length: float = declare(default=0, unit="m", unit_comment="of root segment", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    radius: float = declare(default=0, unit="m", unit_comment="of root segment", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    struct_mass: float = declare(default=0, unit="g", unit_comment="of dry weight", description="", 
                                 min_value="", max_value="", value_comment="", references="", DOI="",
                                 variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    
    # FROM N MODEL
    Nm: float =                 declare(default=1e-4, unit="mol.g-1", unit_comment="of nitrates", description="",
                                        min_value=1e-6, max_value=1e-3, value_comment="", references="", DOI="",
                                        variable_type="input", by="model_nitrogen", state_variable_type="intensive", edit_by="user")
    AA: float =                 declare(default=9e-4, unit="mol.g-1", unit_comment="of amino acids", description="",
                                        min_value=1e-5, max_value=1e-2, value_comment="", references="", DOI="",
                                        variable_type="input", by="model_nitrogen", state_variable_type="intensive", edit_by="user")

    # FROM SHOOT MODEL
    water_root_shoot_xylem: float = declare(default=0., unit="mol.s-1", unit_comment="of water", description="", 
                                            min_value="", max_value="", value_comment="", references="", DOI="",
                                            variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")

    # --- INITIALIZE MODEL STATE VARIABLES ---

    # LOCAL VARIABLES

    # Pools initial values
    xylem_water: float = declare(default=0., unit="mol", unit_comment="of water", description="", 
                                 min_value="", max_value="", value_comment="", references="",  DOI="",
                                 variable_type="state_variable", by="model_water", state_variable_type="extensive", edit_by="user")

    # Water transport processes
    radial_import_water: float = declare(default=0., unit="mol.time_step-1", unit_comment="of water", description="", 
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="state_variable", by="model_water", state_variable_type="extensive", edit_by="user")
    axial_export_water_up: float = declare(default=0., unit="mol.time_step-1", unit_comment="of water", description="",
                                           min_value="", max_value="", value_comment="", references="", DOI="",
                                           variable_type="state_variable", by="model_water", state_variable_type="extensive", edit_by="user")
    axial_import_water_down: float = declare(default=0., unit="mol.time_step-1", unit_comment="of water", description="",
                                             min_value="", max_value="", value_comment="", references="", DOI="",
                                             variable_type="state_variable", by="model_water", state_variable_type="extensive", edit_by="user")

    # SUMMED STATE VARIABLES

    total_xylem_water: float = declare(default=0., unit="mol", unit_comment="of water", description="", 
                                       min_value="", max_value="", value_comment="", references="", DOI="",
                                       variable_type="plant_scale_state", by="model_water", state_variable_type="", edit_by="user")
    xylem_total_pressure: float = declare(default=-0.1e6, unit="Pa", unit_comment="", description="apoplastic pressure in stele at rest, we want the -0.5e6 target to be emerging from water balance", 
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="plant_scale_state", by="model_water", state_variable_type="", edit_by="user")

    # --- INITIALIZES MODEL PARAMETERS ---

    # time resolution
    sub_time_step: int = declare(default=3600, unit="s", unit_comment="", description="MUST be a multiple of base time_step", 
                                 min_value="", max_value="", value_comment="", references="", DOI="",
                                 variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")

    # Water properties
    water_molar_mass: float = declare(default=18, unit="g.mol-1", unit_comment="", description="", 
                                      min_value="", max_value="", value_comment="", references="", DOI="",
                                      variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    water_volumic_mass: float = declare(default=1e6, unit="g.m-3", unit_comment="", description="", 
                                        min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    sap_viscosity: float = declare(default=1.3e6, unit="Pa", unit_comment="", description="", 
                                   min_value="", max_value="", value_comment="", references="", DOI="",
                                   variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    nonN_solutes: float = declare(default=0, unit="mol.m-3", unit_comment="", description="Non N solutes volumic concentration in xylem", 
                                   min_value="", max_value="", value_comment="Null for now, C could be ignored but other K and P minerals should probably be considered", references="", DOI="",
                                   variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    sigma_water: float = declare(default=1., unit="adim", unit_comment="", description="Effective reflection coefficient for osmotic related water transport flux", 
                                   min_value="", max_value="", value_comment="", references="Bauget et al. 2023", DOI="",
                                   variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")

    # Vessel mechanical properties
    xylem_young_modulus: float = declare(default=1e6, unit="Pa", unit_comment="", description="radial elastic modulus of xylem tissues (Has to be superior to initial difference between root and soil)",
                                         min_value="", max_value="1e6", value_comment="", references="Plavcova 1e9 bending modulus woody species", DOI="",
                                         variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    xylem_cross_area_ratio: float = declare(default=0.84 * (0.36 ** 2), unit="adim", unit_comment="", description=" apoplasmic cross-section area ratio * stele radius ratio^2",
                                            min_value="", max_value="", value_comment="from 0.84 * (0.36 ** 2) to account for buffering", references="", DOI="",
                                            variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")

    cortex_water_conductivity: float = declare(default=1e-14 * 1e3, unit="m.s-1.Pa-1", unit_comment="", description="",
                                               min_value="", max_value="", value_comment="", references="", DOI="",
                                               variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    apoplasmic_water_conductivity: float = declare(default=1e-14 * 1e4, unit="m.s-1.Pa-1", unit_comment="", description="",
                                                   min_value="", max_value="", value_comment="", references="", DOI="",
                                                   variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    xylem_tear: float = declare(default=0.99e6, unit="Pa", unit_comment="", description="maximal difference with soil pressure before xylem tearing (absolute, < xylem_young modulus)",
                                min_value="", max_value="", value_comment="very sensitive, stay bellow young modulus", references="", DOI="",
                                variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")

    def __init__(self, g, time_step, **scenario):
        """
        Description :
        This root water model discretized at root segment's scale intends to account for heterogeneous axial and radial water flows observed in the roots (Bauget et al. 2022).
        
        Hypothesis :
        Accounting for heterogeneous water flows would improbe the overall nutrient balance for root hydromineral uptake.
        """
        self.g = g
        self.props = self.g.properties()
        self.time_step = time_step
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step, data=self.props)
        self.vertices = self.g.vertices(scale=self.g.max_scale())

        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)
        self.link_self_to_mtg()

    def post_coupling_init(self):
        self.pull_available_inputs()
        
        # Must be performed after so that self state variables are indeed dicts
        self.init_xylem_water()

        # SPECIFIC HERE, Select real children for collar element (vid == 1).
        # This is mandatory for correct collar-to-tip Hagen-Poiseuille flow partitioning.
        # CHANGE TO TYPE DESCRIPTION!!!
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
        self.total_xylem_water[1] = ((((self.xylem_total_pressure[1] - np.mean(list(self.soil_water_pressure.values()))) / self.xylem_young_modulus) + 1) ** 2) * (
                np.pi * (np.mean(list(self.radius.values())) ** 2) * sum(self.length.values()) * self.xylem_cross_area_ratio * self.water_volumic_mass) / self.water_molar_mass

        sum_volume = sum(self.xylem_volume.values())

        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:
                self.xylem_water[vid] = self.total_xylem_water[1] * self.xylem_volume[vid] / sum_volume

    def post_growth_updating(self):
        """
        Description :
            Extend property dictionnary uppon new element partionning and updates concentrations uppon structural_mass change
        """
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        for vid in self.vertices:
            if vid not in list(self.xylem_water.keys()):
                parent = self.g.parent(vid)
                mass_fraction = self.struct_mass[vid] / (self.struct_mass[vid] + self.struct_mass[parent])
                for prop in self.state_variables:
                    # if intensive, equals to parent
                    if self.__dataclass_fields__[prop].metadata["state_variable_type"] == "intensive":
                        getattr(self, prop).update({vid: getattr(self, prop)[parent]})
                    # if extensive, we need structural mass wise partitioning
                    else:
                        getattr(self, prop).update({vid: getattr(self, prop)[parent] * mass_fraction,
                                                    parent: getattr(self, prop)[parent] * (1 - mass_fraction)})
    
    @state
    def transport_water(self):
        # Using previous time-step flows, we compute current time-step pressure for flows computation

        # Compute the minimal water content for current dimensions under which we would have shearing and cavitation processes
        tearing_total_xylem_water = (((- self.xylem_tear / self.xylem_young_modulus) + 1) ** 2) * (
                  np.pi * (np.mean(list(self.radius.values())) ** 2) * sum(
              self.length.values()) * self.xylem_cross_area_ratio * self.water_volumic_mass) / self.water_molar_mass
        # We compute a target equilibrium water content that will be converged when no transpiration disturbance occur.
        flux_inversion_xylem_water = (((+ 0. / self.xylem_young_modulus) + 1) ** 2) * (
                  np.pi * (np.mean(list(self.radius.values())) ** 2) * sum(
              self.length.values()) * self.xylem_cross_area_ratio * self.water_volumic_mass) / self.water_molar_mass

        # we set collar element the flow provided by shoot model
        potential_transpiration = self.water_root_shoot_xylem[1] * self.time_step
        # condition if potential transpiration is going to lead to a tearing pressure of xylem
        # print(tearing_total_xylem_water, self.total_xylem_water[1] - potential_transpiration + sum(self.radial_import_water.values()))

        sum_volume = sum(self.xylem_volume.values())
        # Loop computing individual segments' water exchange
        for vid in self.vertices:
            if self.struct_mass[vid] > 0:
                # radial exchanges are only hydrostatic-driven for now
                apoplastic_water_import = self.apoplasmic_water_conductivity * (self.soil_water_pressure[vid] - self.xylem_total_pressure[1]) * self.apoplasmic_exchange_surface[vid]

                cross_membrane_water_import = self.cortex_water_conductivity * (
                    self.soil_water_pressure[vid] - self.xylem_total_pressure[1] + self.sigma_water * 8.314 * (273.15 + self.soil_temperature[vid])*(
                        self.soil_solutes[vid] - (
                        (self.Nm[vid] * self.struct_mass[vid] / self.xylem_volume[vid]) + (self.AA[vid] * self.struct_mass[vid] / self.xylem_volume[vid]) + self.nonN_solutes)
                        )
                    ) * self.cortex_exchange_surface[vid]

                self.radial_import_water[vid] = (apoplastic_water_import + cross_membrane_water_import) * self.time_step

                # We also take advantage of this loop to compute the local water content to prevent another for loop over mtg
                self.xylem_water[vid] = self.total_xylem_water[1] * self.xylem_volume[vid] / sum_volume

        total_radial_import_water = sum(self.radial_import_water.values())

        # From potential transpiration and water import fluxes, we compute the actual transpiration if potential is too high
        if self.total_xylem_water[1] - potential_transpiration + total_radial_import_water < tearing_total_xylem_water:
            self.axial_export_water_up[1] = max(0., self.total_xylem_water[1] - tearing_total_xylem_water + sum(self.radial_import_water.values()))
            print(f"\n[Warning] Artificially restraining transpiration at {round(100*self.axial_export_water_up[1]/potential_transpiration, 2)} %")
        else:
            self.axial_export_water_up[1] = potential_transpiration

        # From potential water import fluxes, if they are so important they could lead to pressurisation which is not observed experimentally,
        # to avoid this behavior that is possible with such a discrete model, we converge towards the above precomputed equilibrium
        if total_radial_import_water > 0. and self.total_xylem_water[1] - self.axial_export_water_up[1] + total_radial_import_water > flux_inversion_xylem_water:
            excess_import_water = self.total_xylem_water[1] - potential_transpiration + total_radial_import_water - flux_inversion_xylem_water
            for vid in self.vertices:
                self.radial_import_water[vid] -= excess_import_water * self.radial_import_water[vid] / total_radial_import_water
            total_radial_import_water = sum(self.radial_import_water.values())

        total_delta_water = total_radial_import_water - self.axial_export_water_up[1]
                
        # Finally we compute the axial result of these transpiration fluxes and radial uptake
        # We define "root" as the starting point of the loop below:
        root_gen = self.g.component_roots_at_scale_iter(self.g.root, scale=1)
        root = next(root_gen)

        # We travel in the MTG from the root collar to the tips:
        for vid in pre_order2(self.g, root):
            # We apply the following for all structural mass, because null length element can be support for
            # ramification

            child = self.g.children(vid)

            # if we look at a collar artificial vertex of null lenght, we do nothing
            if vid not in self.collar_skip:

                # We suppose the xylem depletion or filling is evenly reparted during the time step
                delta_water = total_delta_water * self.xylem_water[vid] / self.total_xylem_water[1]

                # For current vertex, compute axial down flow from axial upper flow, radial flow and volume at considered pressure
                # There is no pressure variation effect as water is incompressible

                # If this is a root tip or a non-emerged root segment, there is no down import
                if (vid != 1) and ((len(child) == 0) or (True not in [self.struct_mass[k] > 0 for k in child])):
                    self.axial_import_water_down[vid] = 0

                # if there are children, there is a down import flux
                else:
                    self.axial_import_water_down[vid] = delta_water - self.radial_import_water[vid] + self.axial_export_water_up[vid]

                # For reference, balance between flows is :
                # delta_water = self.radial_import_water[vid] - self.axial_export_water_up[vid] + self.axial_import_water_down[vid]

                # For current vertex's children, provide previous down flow as axial upper flow for children
                if len(child) == 1:
                    # if current vertex is collar, we affect down flow at previously computed collar children
                    if vid == 1 and len(self.collar_children) > 0:
                        HP = [0 for k in self.collar_children]
                        for k in range(len(self.collar_children)):
                            # compute Hagen-Poiseuille coefficient
                            HP[k] = np.pi * (self.radius[self.collar_children[k]]**4) / (8 * self.sap_viscosity)
                        HP_tot = sum(HP)
                        for k in range(len(self.collar_children)):
                            self.axial_export_water_up[self.collar_children[k]] = (HP[k] / HP_tot) * self.axial_import_water_down[vid]
                    # else, if there is only one child, the entire flux is applied
                    else:
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

        # Computed here as it is required by pressure computations
        self.total_xylem_water[1] += total_delta_water

        # Finally, we assume pressure homogeneity and compute the resulting pressure for the next time_step
        self.xylem_total_pressure[1] = self.xylem_young_modulus * (
                    (((self.total_xylem_water[1] * self.water_molar_mass) / (
                        np.pi * (np.mean(list(self.radius.values())) ** 2) * sum(self.length.values()) *
                            self.xylem_cross_area_ratio * self.water_volumic_mass)) ** 0.5) - 1) + np.mean(
                            list(self.soil_water_pressure.values()))
