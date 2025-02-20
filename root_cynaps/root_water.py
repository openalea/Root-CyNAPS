import numpy as np
from numba import njit
from openalea.mtg.traversal import pre_order2
from dataclasses import dataclass
import inspect

from metafspm.component import Model, declare
from metafspm.component_factory import *

from scipy.integrate import solve_ivp


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
    C_solutes_soil: float = declare(default=0., unit="mol.m-3", unit_comment="", description="Total soil solute volumic concentration", 
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
    type: str = declare(default="Normal_root_after_emergence", unit="", unit_comment="", description="Example segment type provided by root growth model", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    
    # FROM N MODEL
    xylem_Nm: float =                 declare(default=1e-4, unit="mol.g-1", unit_comment="of nitrates", description="",
                                        min_value=1e-6, max_value=1e-3, value_comment="", references="", DOI="",
                                        variable_type="input", by="model_nitrogen", state_variable_type="intensive", edit_by="user")
    xylem_AA: float =                 declare(default=9e-4, unit="mol.g-1", unit_comment="of amino acids", description="",
                                        min_value=1e-5, max_value=1e-2, value_comment="", references="", DOI="",
                                        variable_type="input", by="model_nitrogen", state_variable_type="intensive", edit_by="user")
    xylem_struct_mass: float =        declare(default=0, unit="g", unit_comment="of xylem structural mass", description="",
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
    sigma_water: float = declare(default=0.85, unit="adim", unit_comment="", description="Effective reflection coefficient for osmotic related water transport flux", 
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

        # Initial solved function lookup for efficient solver feeding
        self.water_solver_kwargs = [param.name for param in inspect.signature(root_water_dynamics).parameters.values() 
                                    if param.name not in {"t", "y", "adjacency", "K_axial"}]
        self.water_solver_inputs_names = [name for name in self.water_solver_kwargs if isinstance(getattr(self, name), dict)]
        self.water_solver_params = {name: getattr(self, name) for name in self.water_solver_kwargs if not isinstance(getattr(self, name), dict)}

    def post_coupling_init(self):
        self.pull_available_inputs()
        
        # Must be performed after so that self state variables are indeed dicts
        self.init_xylem_water()

        # SPECIFIC HERE, Select real children for collar element (vid == 1).
        # This is mandatory for correct collar-to-tip Hagen-Poiseuille flow partitioning.
        # CHANGE TO TYPE DESCRIPTION!!!
        self.collar_children, self.collar_skip = [], []
        for vid in self.vertices:
            children = self.g.children(vid)
            if self.type[vid] in ('Support_for_seminal_root', 'Support_for_adventitious_root') and children:
                self.collar_skip += [vid]
                self.collar_children += [k for k in children if self.type[k] not in ('Support_for_seminal_root', 'Support_for_adventitious_root')]
        
    def init_xylem_water(self):
        # We initialize the xylem water content heterogeneity from initial soil - root pressure gradients

        for vid in self.vertices:
            # if root segment emerged
            if self.struct_mass[vid] > 0:
                self.xylem_water[vid] = ((((self.soil_water_pressure[vid] - self.xylem_total_pressure[1])/self.xylem_young_modulus) + 1)**2)*(
                np.pi*(self.radius[vid]**2)*self.length[vid]*self.xylem_cross_area_ratio*self.water_volumic_mass)/self.water_molar_mass

                self.total_xylem_water[1] += self.xylem_water[vid]


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
    

    @rate
    def transport_water_heterogeneous(self):
        # We map vid to matrix indices to avoid any mistake when updating the dictionnaries from numpy arrays
        vid_to_indice = {vid: i for i, vid in enumerate(self.struct_mass.keys())}
        n = len(vid_to_indice)

        adjacency = np.zeros((n, n))

        for vid, i in vid_to_indice.items():
            if vid == 1: # If this is collar we assign precomputed children
                children = self.collar_children
            elif vid not in self.collar_skip:
                children = self.g.children(vid)
            else:
                continue
            # Then we edit the adjacency matrix
            for child in children:
                adjacency[i, vid_to_indice[child]] = 1

        # Parameters don't change but inputs are to be retrieved
        kwargs = self.water_solver_params
        kwargs.update({name: np.array(list(getattr(self, name).values())) for name in self.water_solver_inputs_names})

        # Specific inputs are finally added here
        kwargs["adjacency"] = adjacency
        kwargs["K_axial"] = np.where(kwargs["length"] > 0., np.pi * (kwargs["radius"]**4) / (8 * self.sap_viscosity * kwargs["length"]), 0.)

        # Solver call preparation
        W_init = np.array(list(getattr(self, "xylem_water").values()))

        n = len(W_init)

        y0 = np.concatenate((W_init, np.zeros_like(W_init), np.zeros_like(W_init), np.zeros_like(W_init)))

        # Solve ODE system with adaptive time stepping and step rejection
        sol = solve_ivp(
            wrapped_root_water_dynamics, t_span=(0, self.time_step), y0=y0,
            args=(kwargs,), method="RK45", dense_output=True,
            events=lambda t, y, unused_kwargs: water_violation_event(t, y, xylem_water_min=0.5*kwargs["xylem_volume"], n=n)  # Step rejection event
        )

        # Only final root state is of insterest when interacting with other modules
        W_solutions = sol.y[:n, :][:, -1] 
        q_axial_out_solutions = sol.y[n:2*n, :][:, -1]
        q_axial_in_solutions = sol.y[2*n:3*n, :][:, -1]
        q_radial_solutions = sol.y[3*n:, :][:, -1]

        # expected = W_init + q_axial_in_solutions - q_axial_out_solutions + q_radial_solutions 
        # obtained = W_solutions
        # assert False in (obtained == expected)

        print(W_solutions, q_axial_out_solutions, q_axial_in_solutions, q_radial_solutions)

        

    #@rate
    def transport_water(self):

        # Using previous time-step flows, we compute current time-step pressure for flows computation
        numpy_args = ["radius", "length", "xylem_volume", "soil_water_pressure", "apoplasmic_exchange_surface", "soil_temperature", "C_solutes_soil", "xylem_Nm", "xylem_AA", "xylem_struct_mass", "cortex_exchange_surface"]
        
        args = {arg: np.array(list(getattr(self, arg).values())) for arg in numpy_args}
        states = [self.water_root_shoot_xylem[1]]
        states += [args[arg] for arg in ("soil_water_pressure", "apoplasmic_exchange_surface", "soil_temperature", "C_solutes_soil", "xylem_Nm", "xylem_struct_mass", "xylem_volume", "xylem_AA", "cortex_exchange_surface")]

        class_params = ["apoplasmic_water_conductivity", "cortex_water_conductivity", "sigma_water", "nonN_solutes"]

        params = [getattr(self, param) for param in class_params]

        norm_soil_pressure = np.sum((args["radius"]**2) * args["length"] * args["soil_water_pressure"]) / np.sum((args["radius"]**2) * args["length"])
        
        norm_soil_pressure_square = np.sum((args["radius"]**2) * args["length"] * (args["soil_water_pressure"]**2)) / np.sum((args["radius"]**2) * args["length"])

        A = - self.xylem_young_modulus + norm_soil_pressure
        B = 4 * self.xylem_young_modulus * norm_soil_pressure + ((norm_soil_pressure)**2) - norm_soil_pressure_square
        C = (self.xylem_young_modulus**2) * self.water_molar_mass / (np.pi * self.xylem_cross_area_ratio * self.water_volumic_mass * np.sum((args["radius"]**2) * args["length"]))
        
        params.append(A)
        params.append(B)
        params.append(C)
        
        y0 = np.concatenate([[self.total_xylem_water[1]], np.zeros_like(args["xylem_struct_mass"])])

        # Solve the system
        sol = solve_ivp(solve_water_transport, (0, self.time_step), y0, args=states + params)
        
        # Extract results
        self.total_xylem_water[1] = sol.y[0][-1]
        self.radial_import_water.update(dict(zip(self.xylem_water.keys(), sol.y[1:][:, -1])))

        # Then knowing the system convergence we compute the related local water content
        if B + C * self.total_xylem_water[1] > 0:
            self.xylem_total_pressure[1] = A + (B + C*self.total_xylem_water[1])**0.5
        else:
            self.xylem_total_pressure[1] = A

        xylem_water = ((((args["soil_water_pressure"] - self.xylem_total_pressure[1])/self.xylem_young_modulus) + 1)**2)*(
            np.pi*(args["radius"]**2)*args["length"]*self.xylem_cross_area_ratio*self.water_volumic_mass)/self.water_molar_mass
        
        print(self.total_xylem_water[1], np.sum(xylem_water))

        # But we don't uptake the dict now to compute the delta
        new_xylem_water = dict(zip(self.xylem_water.keys(), xylem_water))

                
        # Finally we compute the axial result of the transpiration flux downward propagation
        self.axial_export_water_up[1] = self.water_root_shoot_xylem[1] * self.time_step

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

                # For current vertex, compute axial down flow from axial upper flow, radial flow and volume at considered pressure
                # There is no pressure variation effect as water is incompressible

                # If this is a root tip or a non-emerged root segment, there is no down import
                if (vid != 1) and ((len(child) == 0) or (True not in [self.struct_mass[k] > 0 for k in child])):
                    self.axial_import_water_down[vid] = 0

                # if there are children, there is a down import flux
                else:
                    self.axial_import_water_down[vid] = (new_xylem_water[vid] - self.xylem_water[vid]) - self.radial_import_water[vid] * self.time_step + self.axial_export_water_up[vid]
                
                # We also actualize the new content here to prevent further dict calls
                self.xylem_water[vid] = new_xylem_water[vid]

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

#@njit
def solve_water_transport(t, y, *args):

    total_xylem_water = y[0]
    Tf_water = y[1:]

    # Unpack states
    (water_root_shoot_xylem, soil_water_pressure, apoplasmic_exchange_surface, soil_temperature, C_solutes_soil, xylem_Nm, xylem_struct_mass, xylem_volume, xylem_AA, cortex_exchange_surface, 
    # Then parameters
    apoplasmic_water_conductivity, cortex_water_conductivity, sigma_water, nonN_solutes, A, B, C) = args

    if B + C * total_xylem_water > 0:
        xylem_total_pressure = A + (B + C*total_xylem_water)**0.5
    else:
        xylem_total_pressure = A

    dradial_water_dt = _radial_import_water(xylem_total_pressure, soil_water_pressure, soil_temperature, 
                         C_solutes_soil, xylem_Nm, xylem_AA, xylem_struct_mass, xylem_volume, 
                         apoplasmic_exchange_surface, cortex_exchange_surface, 
                         (apoplasmic_water_conductivity, cortex_water_conductivity, sigma_water, nonN_solutes))

    dtotal_water_dt = np.sum(dradial_water_dt) - water_root_shoot_xylem

    
    return [dtotal_water_dt] + list(dradial_water_dt)


#@njit
def root_water_dynamics(t, y, 
                        # Fixed input root states...
                        # (Axial flux related)
                        adjacency, water_root_shoot_xylem, radius, length, soil_water_pressure, K_axial, 
                        # (Radial uptake related)
                        apoplasmic_exchange_surface, soil_temperature, C_solutes_soil, xylem_Nm, xylem_struct_mass, xylem_volume, xylem_AA, cortex_exchange_surface, 
                        
                        # Then parameters...
                        # (Pressure related)
                        xylem_young_modulus, water_molar_mass, xylem_cross_area_ratio, water_volumic_mass, 
                        # (Radial uptake related)
                        apoplasmic_water_conductivity, cortex_water_conductivity, sigma_water, nonN_solutes):
    """
    Computes the time derivatives of xylem pressure and returns fluxes.

    t: Current time (needed for solve_ivp, but not used here)
    y: Solved states provided as input
    args: Fixed boundary conditions of the model, to be provided in a specific order and not with keywords to remain numba compatible

    Returns:
    dpsi_dt: Time derivative of xylem pressures
    """
    # # Unpack fixed input root states...
    # # (Axial flux related)
    # (adjacency, water_root_shoot_xylem, radius, length, soil_water_pressure, K_axial, 
    # # (Radial uptake related)
    # apoplasmic_exchange_surface, soil_temperature, C_solutes_soil, xylem_Nm, xylem_struct_mass, xylem_volume, xylem_AA, cortex_exchange_surface, 
    
    # # Then parameters...
    # # (Pressure related)
    # xylem_young_modulus, water_molar_mass, xylem_cross_area_ratio, water_volumic_mass, 
    # # (Radial uptake related)
    # apoplasmic_water_conductivity, cortex_water_conductivity, sigma_water, nonN_solutes) = args

    # Unpack solved variables...
    n = len(xylem_struct_mass)
    xylem_water = y[:n]
    q_axial_out_previous = y[n:2*n]
    q_axial_in_previous = y[2*n:3*n]
    q_radial_previous = y[3*n:]

    # Computing local water pressure
    xylem_water_pressure = np.where(length > 0, xylem_young_modulus * (
        ((xylem_water * water_molar_mass / (np.pi * (radius**2) * length * xylem_cross_area_ratio * water_volumic_mass) )**0.5) - 1
                                                  ) + soil_water_pressure, 0.)

    # Compute axial flux using matrix operations (Hagen-Poiseuille) and Δψ between connected segments
    q_axial_out = K_axial * (adjacency.T @ xylem_water_pressure - xylem_water_pressure)

    # Collar element export wouldn't have been computed in previous step, so we set this boundary condition here
    q_axial_out[0] = water_root_shoot_xylem[0]

    # Compute total inflow per node
    # Sum of children fluxes, when there are no children, a zero downward boundary condition is set implicitely here
    q_axial_in = adjacency @ q_axial_out
      
    # Compute radial flux
    q_radial = _radial_import_water(xylem_water_pressure, soil_water_pressure, soil_temperature, 
                         C_solutes_soil, xylem_Nm, xylem_AA, xylem_struct_mass, xylem_volume, 
                         apoplasmic_exchange_surface, cortex_exchange_surface, 
                         (apoplasmic_water_conductivity, cortex_water_conductivity, sigma_water, nonN_solutes))

    # Compute pressure derivatives using water balance
    dxylem_water_dt = q_axial_in - q_axial_out + q_radial  # Element-wise division

    # Stack the outputs so we can return both pressure changes and fluxes
    return np.concatenate((dxylem_water_dt, q_axial_out, q_axial_in, q_radial))


def wrapped_root_water_dynamics(t, y, *kwargs_in_args):
    return root_water_dynamics(t, y, **kwargs_in_args[0])

#@njit
def _radial_import_water(xylem_water_pressure, soil_water_pressure, soil_temperature, 
                         C_solutes_soil, xylem_Nm, xylem_AA, xylem_struct_mass, xylem_volume, 
                         apoplasmic_exchange_surface, cortex_exchange_surface, params):
    
    apoplasmic_water_conductivity, cortex_water_conductivity, sigma_water, nonN_solutes = params

    apoplastic_water_import = apoplasmic_water_conductivity * (soil_water_pressure - xylem_water_pressure) * apoplasmic_exchange_surface
    
    # xylem_Nm_volumic = np.divide(xylem_Nm * xylem_struct_mass, xylem_volume, where=xylem_volume > 0., out=np.zeros_like(xylem_volume))
    # xylem_AA_volumic = np.divide(xylem_AA * xylem_struct_mass, xylem_volume, where=xylem_volume > 0., out=np.zeros_like(xylem_volume))
    
    #If njit
    xylem_Nm_volumic = np.where(xylem_volume > 0., xylem_Nm * xylem_struct_mass / xylem_volume, 0.)
    xylem_AA_volumic = np.where(xylem_volume > 0., xylem_AA * xylem_struct_mass / xylem_volume, 0.)

    cross_membrane_water_import = cortex_water_conductivity * (
                soil_water_pressure - xylem_water_pressure - sigma_water * 8.314 * (273.15 + soil_temperature)*(C_solutes_soil -
                    xylem_Nm_volumic + xylem_AA_volumic + nonN_solutes)
                ) * cortex_exchange_surface
    
    return apoplastic_water_import + cross_membrane_water_import


def water_violation_event(t, y, xylem_water_min, n):
    """
    Event function to detect when water content W goes below W_min.
    If W < W_min significantly, the solver should reject the step.

    t: Current time step
    y: Current state vector (contains xylem water content W)
    W_min: Minimum allowable water content per segment
    n: Number of root segments

    Returns:
    A small positive value when W is okay, negative when W is too low.
    """

    xylem_water = y[:n]  # Extract water content from the state vector
    W_diff = xylem_water - xylem_water_min  # Difference between current W and W_min

    # If any W is significantly below W_min, trigger a hard rejection event
    if np.any(W_diff < -0.05 * xylem_water_min):  # Customize threshold for "too far"
        return -1.0  # Hard rejection → solver must retry

    # If W is slightly below W_min, trigger a soft step reduction
    return np.min(W_diff)

# Configure event properties
water_violation_event.terminal = True  # Stop and retry step when triggered
water_violation_event.direction = -1   # Detect only when approaching from positive values towards negative
