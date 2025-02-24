import numpy as np
from numba import njit
from openalea.mtg.traversal import pre_order2
from dataclasses import dataclass
import inspect
from time import time

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
    xylem_pressure: float = declare(default=-0.1e6, unit="Pa", unit_comment="", description="apoplastic pressure in stele at rest, we want the -0.5e6 target to be emerging from water balance", 
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="intensive", edit_by="user")

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
        self.minimal_water_fraction = ((-0.9e6 / self.xylem_young_modulus) + 1)**2 # Here we set the maximum difference with soil potential to 0.9 MPa

        self.tp_ct = 0

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
                self.xylem_water[vid] = ((((self.soil_water_pressure[vid] - self.xylem_pressure[vid])/self.xylem_young_modulus) + 1)**2)*(
                np.pi*(self.radius[vid]**2)*self.length[vid]*self.xylem_cross_area_ratio*self.water_volumic_mass)/self.water_molar_mass


    def post_growth_updating(self):
        """
        Description :
            Extend property dictionnary uppon new element partionning and updates concentrations uppon structural_mass change
        """
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        for vid in self.vertices:
            if vid not in list(self.xylem_water.keys()) or (self.xylem_water[vid] == 0. and self.struct_mass[vid] > 0.):
                parent = self.g.parent(vid)
                mass_fraction = self.struct_mass[vid] / (self.struct_mass[vid] + self.struct_mass[parent])
                for prop in self.state_variables:
                    # if intensive, equals to parent
                    if self.__dataclass_fields__[prop].metadata["state_variable_type"] == "intensive":
                        getattr(self, prop).update({vid: getattr(self, prop)[parent]})
                    # if extensive, we need structural mass wise partitioning
                    elif prop != "xylem_water":
                        getattr(self, prop).update({vid: getattr(self, prop)[parent] * mass_fraction,
                                                    parent: getattr(self, prop)[parent] * (1 - mass_fraction)})
                        
                # Once pressure has been set equal to parent
                self.xylem_water[vid] = ((((self.soil_water_pressure[vid] - self.xylem_pressure[vid])/self.xylem_young_modulus) + 1)**2)*(
                                    np.pi*(self.radius[vid]**2)*self.length[vid]*self.xylem_cross_area_ratio*self.water_volumic_mass)/self.water_molar_mass
    

    @rate
    def transport_water(self):
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
        kwargs["K_axial"] = np.where(kwargs["length"] > 0., np.pi * ((0.2*kwargs["radius"])**4) / (8 * self.sap_viscosity * kwargs["length"]), 0.)

        # Solver call preparation
        W_init = np.array(list(getattr(self, "xylem_water").values()))
            
        # Solve ODE system with adaptive time stepping and step rejection
        sol = solve_ivp(
            wrapped_root_water_dynamics, t_span=(0, self.time_step), y0=W_init,
            args=(kwargs,), method="Radau", # Try LSODA instead of RK45 might speed up / RK45 / Radau
            events=lambda t, y, unused_kwargs: water_violation_event(t, y, 
                                                xylem_water_min=self.minimal_water_fraction*kwargs["xylem_volume"]*self.water_volumic_mass/self.water_molar_mass, n=n),
            jac=lambda t, y, jac_kwargs: root_water_jacobian_derivatives(t, y, **jac_kwargs)
            )

        # Only final root state is of insterest when interacting with other modules
        W_solutions = sol.y[:, -1]

        if np.any(W_solutions < 0.):
            print(True)

        # Update state variables dictionnaries in mtg properties
        self.xylem_water.update(dict(zip(vid_to_indice.keys(), W_solutions)))

        # Recompute fluxes at outputed times
        q_axial_out_solutions, q_axial_in_solutions, q_radial_solutions, xylem_pressure = root_water_derivatives(xylem_water=W_solutions, **kwargs)        
        self.axial_export_water_up.update(dict(zip(vid_to_indice.keys(), q_axial_out_solutions)))
        self.axial_import_water_down.update(dict(zip(vid_to_indice.keys(), q_axial_in_solutions)))
        self.radial_import_water.update(dict(zip(vid_to_indice.keys(), q_radial_solutions)))
        self.xylem_pressure.update(dict(zip(vid_to_indice.keys(), xylem_pressure)))

        self.tp_ct += 1


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

    # Unpack solved variables...
    xylem_water = y

    q_axial_out, q_axial_in, q_radial, _ = root_water_derivatives(xylem_water, 
                            adjacency, water_root_shoot_xylem, radius, length, soil_water_pressure, K_axial, 
                            apoplasmic_exchange_surface, soil_temperature, C_solutes_soil, xylem_Nm, xylem_struct_mass, xylem_volume, xylem_AA, cortex_exchange_surface, 
                            xylem_young_modulus, water_molar_mass, xylem_cross_area_ratio, water_volumic_mass, 
                            apoplasmic_water_conductivity, cortex_water_conductivity, sigma_water, nonN_solutes)

    dxylem_water_dt = q_axial_in - q_axial_out + q_radial

    if np.any(np.isnan(dxylem_water_dt)):
        print("breakpoint nan needed")
    
    elif np.any(np.isinf(dxylem_water_dt)):
        print("breakpoint inf needed")

    return dxylem_water_dt


def wrapped_root_water_dynamics(t, y, *kwargs_in_args):
    return root_water_dynamics(t, y, **kwargs_in_args[0])


#@njit
def root_water_derivatives(xylem_water,
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
    # Computing local water pressure
    xylem_water_pressure = np.where(xylem_water > 0, xylem_young_modulus * (
        ((xylem_water * water_molar_mass / (np.pi * (radius**2) * length * xylem_cross_area_ratio * water_volumic_mass) )**0.5) - 1
                                                  ) + soil_water_pressure, 0.)

    # Compute axial flux using matrix operations (Hagen-Poiseuille) and Δψ between connected segments
    q_axial_out = np.where(xylem_water > 0, K_axial * (xylem_water_pressure - adjacency.T @ xylem_water_pressure), 0) # current - parent pressure

    # Collar element export wouldn't have been computed in previous step, so we set this boundary condition here
    q_axial_out[0] = water_root_shoot_xylem[0]

    # Compute total inflow per node
    # Sum of children fluxes, when there are no children, a zero downward boundary condition is set implicitely here
    q_axial_in = adjacency @ q_axial_out
      
    # Compute radial flux
    q_radial = np.where(xylem_water > 0, _radial_import_water(xylem_water_pressure, soil_water_pressure, soil_temperature, 
                         C_solutes_soil, xylem_Nm, xylem_AA, xylem_struct_mass, xylem_volume, 
                         apoplasmic_exchange_surface, cortex_exchange_surface, 
                         (apoplasmic_water_conductivity, cortex_water_conductivity, sigma_water, nonN_solutes)), 0)

    # return derivatives for water balance
    return q_axial_out, q_axial_in, q_radial, xylem_water_pressure
    
#@njit
def root_water_jacobian_derivatives(t, y,
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

    n = len(y)

    # Here soil water pressure is considered as constant thus
    dpsi_dwater = np.where(y > 0, xylem_young_modulus * (
        (water_molar_mass / (np.pi * (radius**2) * length * xylem_cross_area_ratio * water_volumic_mass) )**0.5) / (2 * (y**0.5)), 0.)

    J_diag = (- (K_axial * dpsi_dwater) # substracting outflow related to i
              + (- dpsi_dwater * (apoplasmic_water_conductivity * apoplasmic_exchange_surface + cortex_water_conductivity * cortex_exchange_surface))) # adding radial component related to i

    J_offdiag = adjacency * (- K_axial * dpsi_dwater)[:, np.newaxis].T # for each children element j, we add to i the related inflow (negative relative to j)

    # Construct full Jacobian matrix
    J = np.eye(n) * J_diag + J_offdiag - np.diag(np.sum(J_offdiag, axis=1))  # Ensure balance by adding what is lost by j off diagonal on element i (sum of down flows, -*- indeed positive)

    # return the full jacobian matrix
    return  J


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
                soil_water_pressure - xylem_water_pressure - sigma_water * 8.314 * (273.15 + soil_temperature) * (C_solutes_soil -
                    xylem_Nm_volumic + xylem_AA_volumic + nonN_solutes)
                ) * cortex_exchange_surface
    
    return apoplastic_water_import + cross_membrane_water_import

@njit
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
    xylem_water = y  # Extract water content from the state vector
    norm_W_diff = (xylem_water - xylem_water_min) / (xylem_water_min + 1e-12) # Difference between current W and W_min

    if np.all(norm_W_diff[xylem_water != 0] > 0.5):  
        return 1  # No issue, return a positive value far from 0 to avoid unnecessary step rejections
    
    elif np.all(xylem_water[xylem_water != 0] > 0.):
        return np.min(norm_W_diff[xylem_water != 0])

    #  we reject any too low values with a hard stop to prevent reaching negative values
    else:
        # print("rejection")
        return -1

# Configure event properties
water_violation_event.terminal = True  # Stop and retry step when triggered
water_violation_event.direction = -1   # Detect only when approaching from positive values towards negative
