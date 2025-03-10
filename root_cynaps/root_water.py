import numpy as np
from numba import njit
from openalea.mtg.traversal import pre_order2
from dataclasses import dataclass
import inspect
from openalea.mtg.traversal import post_order2, pre_order2

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
    soil_temperature: float = declare(default=7.8, unit="°C", unit_comment="", description="soil temperature in contact with roots",
                                        min_value="", max_value="", value_comment="Derived from Swinnen et al. 1994 C inputs, estimated from a labelling experiment starting 3rd of March, with average temperature at 7.8 °C", references="Swinnen et al. 1994", DOI="",
                                        variable_type="input", by="model_temperature", state_variable_type="", edit_by="user")

    # FROM ANATOMY MODEL
    cortex_exchange_surface: float = declare(default=0., unit="m2", unit_comment="", description="", 
                                             min_value="", max_value="", value_comment="", references="", DOI="",
                                             variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    xylem_vessel_radii: float = declare(default=0., unit="m", unit_comment="", description="list of individual xylem radius, also providing their numbering", 
                                             min_value="", max_value="", value_comment="", references="", DOI="",
                                             variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    xylem_volume: float = declare(default=0, unit="m3", unit_comment="", description="xylem volume for water transport between elements", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    kr_symplasmic_water: float = declare(default=1., unit="mol.s-1.Pa-1", unit_comment="", description="Symplasmic water conductance of all cell layer contribution, including transmembrane and plasmodesmata resistance", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    kr_apoplastic_water: float = declare(default=1., unit="mol.s-1.Pa-1", unit_comment="", description="Apolastic water conductance including the endoderm differentiation blocking this pathway. Considering xylem volume to be equivalent to whole stele apoplasm, we only account for the cumulated resistance of cortex and epidermis cell wals.", 
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

    # FROM SHOOT MODEL
    water_root_shoot_xylem: float = declare(default=0., unit="mol.s-1", unit_comment="of water", description="Transpiration related flux at collar", 
                                            min_value="", max_value="", value_comment="", references="", DOI="",
                                            variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    xylem_pressure_collar: float = declare(default=-1e6, unit="Pa", unit_comment="", description="Water potential at collar", 
                                            min_value="", max_value="", value_comment="", references="", DOI="",
                                            variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")

    # --- INITIALIZE MODEL STATE VARIABLES ---

    # LOCAL VARIABLES

    # Pools initial values
    xylem_water: float = declare(default=0, unit="mol", unit_comment="of water", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    xylem_pressure_in: float = declare(default=-0.1e6, unit="Pa", unit_comment="", description="apoplastic pressure in stele at rest, we want the -0.5e6 target to be emerging from water balance", 
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    xylem_pressure_out: float = declare(default=-0.1e6, unit="Pa", unit_comment="", description="apoplastic pressure in stele at rest, we want the -0.5e6 target to be emerging from water balance", 
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    
    # Conductance values
    K: float = declare(default=0, unit="mol.Pa-1.s-1", unit_comment="", description="axial root segment conductance", 
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    Keq: float = declare(default=0, unit="mol.Pa-1.s-1", unit_comment="", description="Equivalent conductance of the current root segment considering its position in the root system", 
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")

    # Water properties
    # sap_viscosity: float = declare(default=1.003e-3, unit="Pa.s", unit_comment="", description="Viscosity at 20°C", 
    #                                min_value="0.535e-3", max_value="1.753e-3", value_comment="", references="", DOI="",
    #                                variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")

    # Water transport processes
    radial_import_water: float = declare(default=0., unit="mol.time_step-1", unit_comment="of water", description="", 
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    axial_export_water_up: float = declare(default=0., unit="mol.time_step-1", unit_comment="of water", description="",
                                           min_value="", max_value="", value_comment="", references="", DOI="",
                                           variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    axial_import_water_down: float = declare(default=0., unit="mol.time_step-1", unit_comment="of water", description="",
                                             min_value="", max_value="", value_comment="", references="", DOI="",
                                             variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")

    # --- INITIALIZES MODEL PARAMETERS ---

    collar_flux_provided: bool = declare(default=False, unit="adim", unit_comment="", description="Option if collar flux is provided by input data", 
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


    def post_coupling_init(self):
        self.pull_available_inputs()
        

        # SPECIFIC HERE, Select real children for collar element (vid == 1).
        # This is mandatory for correct collar-to-tip Hagen-Poiseuille flow partitioning.
        self.collar_children, self.collar_skip = [], []
        for vid in self.vertices:
            children = self.g.children(vid)
            if self.type[vid] in ('Support_for_seminal_root', 'Support_for_adventitious_root') and children:
                self.collar_skip += [vid]
                self.collar_children += [k for k in children if self.type[k] not in ('Support_for_seminal_root', 'Support_for_adventitious_root')]

    @potential
    @rate
    def _K(self, soil_temperature, length, xylem_vessel_radii):
        sap_viscosity = (2.414 * 1e-5) * 10 ** (247.8 / (273.15 + soil_temperature - 140))
        print(sap_viscosity)
        return sum((np.pi * (vessel_radius ** 4) / (8 * sap_viscosity * length)) for vessel_radius in xylem_vessel_radii)

    @actual
    @rate
    def water_transport(self):
        """Compute the water potential and fluxes of each segment
        
        For each vertex of the root, compute :
            - the water potential (:math:`\psi_{\\text{out}}`) at the base;
            - the water potential (:math:`\psi_{\\text{in}}`) at the end;
            - the water flux (`J`) at the base;
            - the lateral water flux (`j`) entering the segment.

        The vertex base is the side toward the basal direction, the vertex end is the one toward the root tip.

        :Algorithm:

            The algorithm has two stages:

                - First, on each segment, an equivalent conductance is computed in post_order (children before parent).
                - Finally, the water flux and potential are computed in pre order (parent then children).

        .. note::
            Here :math:`\psi` are the hydrostatic water potential i.e. the hydrostatic pressure.
            There are no osmotic components.
        """
        
        g = self.g # To prevent repeated MTG lookups

        # Select the base of the root
        root = next(g.component_roots_at_scale_iter(g.root, scale=1))

        # Equivalent conductance computation from tip to collar
        for v in post_order2(g, root):
            if self.struct_mass[v] > 0.:
                if v == root:
                    children = self.collar_children
                else:
                    children = [child for child in g.children(v) if self.struct_mass[child] > 0.]
                
                r = 1./(self.kr_symplasmic_water[v] + self.kr_apoplastic_water[v] + sum(self.Keq[cid] for cid in children))
                R = 1./self.K[v]
                self.Keq[v] = 1. / (r + R)

        # Water flux and water potential computation from collar to tips
        for v in pre_order2(g, root):
            # Compute psi according to Millman theorem, then compute radial flux
            if self.struct_mass[v] > 0:
                if v in self.collar_children:
                    parent = 1
                    brothers = self.collar_children
                else:
                    parent = g.parent(v)
                    brothers = [sibling for sibling in g.children_iter(parent) if self.struct_mass[sibling] > 0.]

                if v == root:
                    children = self.collar_children
                else:
                    children = [child for child in g.children_iter(v) if self.struct_mass[child] > 0.]

                Keq_brothers = sum( self.Keq[cid] for cid in brothers)
                Keq_children = sum( self.Keq[cid] for cid in children)

                if parent is None:
                    self.xylem_pressure_out[v] = self.xylem_pressure_collar[1]

                    # If collar flux is provided by the shoot model
                    if self.collar_flux_provided:
                        self.axial_export_water_up[v] = self.water_root_shoot_xylem[1]
                    # Else we compute the flux according to the Haggen-Poiseuille conductance of
                    else:
                        self.axial_export_water_up[v] = self.K[v] * (self.xylem_pressure_in[v] - self.xylem_pressure_out[v])

                else:
                    self.xylem_pressure_out[v] = self.xylem_pressure_in[parent]
                    self.axial_export_water_up[v] = (self.axial_export_water_up[parent] - self.radial_import_water[parent]) * ( self.Keq[v] / Keq_brothers )
                
                k_radial = self.kr_symplasmic_water[v] + self.kr_apoplastic_water[v]

                self.xylem_pressure_in[v] = (self.K[v] * self.xylem_pressure_out[v] + self.soil_water_pressure[v] * (k_radial + Keq_children)) / (k_radial + self.K[v] + Keq_children)
                self.radial_import_water[v] = (self.soil_water_pressure[v] - self.xylem_pressure_in[v]) * k_radial

                # Computed to avoid children iteration when needed by other modules
                self.axial_import_water_down[v] = self.axial_export_water_up[v] - self.radial_import_water[v]


    @state
    def _xylem_water(self, xylem_volume):
        return xylem_volume * 1e6 / 18
