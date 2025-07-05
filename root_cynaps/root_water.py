import numpy as np
from numba import njit
from openalea.mtg.traversal import pre_order2
from dataclasses import dataclass
import inspect
from openalea.mtg.traversal import post_order2, pre_order2

from openalea.metafspm.component import Model, declare
from openalea.metafspm.component_factory import *

from scipy.sparse import csc_matrix, linalg


debug = True

@dataclass
class RootWaterModel(Model):


    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

    # FROM SOIL MODEL
    soil_water_pressure: float = declare(default=0., unit="Pa", unit_comment="of water", description="", 
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="input", by="model_soil", state_variable_type="", edit_by="user")
    soil_temperature: float = declare(default=7.8, unit="°C", unit_comment="", description="soil temperature in contact with roots",
                                        min_value="", max_value="", value_comment="Derived from Swinnen et al. 1994 C inputs, estimated from a labelling experiment starting 3rd of March, with average temperature at 7.8 °C", references="Swinnen et al. 1994", DOI="",
                                        variable_type="input", by="model_temperature", state_variable_type="", edit_by="user")
    Cv_solute_soil: float = declare(default=0., unit="mol.m-3", unit_comment="of total solutes", description="Total solute concentration in soil", 
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="input", by="model_soil", state_variable_type="", edit_by="user")

    # FROM ANATOMY MODEL
    xylem_vessel_radii: float = declare(default=0., unit="m", unit_comment="", description="list of individual xylem vessel radius, also providing their numbering", 
                                             min_value="", max_value="", value_comment="", references="", DOI="",
                                             variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    phloem_vessel_radii: float = declare(default=0., unit="m", unit_comment="", description="list of individual phloem vessel radius, also providing their numbering", 
                                             min_value="", max_value="", value_comment="", references="", DOI="",
                                             variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    xylem_volume: float = declare(default=0, unit="m3", unit_comment="", description="xylem volume for water transport between elements", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    phloem_volume: float = declare(default=0, unit="m3", unit_comment="", description="phloem volume for water transport between elements", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    kr_symplasmic_water_xylem: float = declare(default=1., unit="m3.s-1.Pa-1", unit_comment="", description="Effective Symplasmic water conductance of all cell layer contribution, including transmembrane and plasmodesmata resistance", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    kr_apoplastic_water_xylem: float = declare(default=1., unit="m3.s-1.Pa-1", unit_comment="", description="Effective Apolastic water conductance including the endoderm differentiation blocking this pathway. Considering xylem volume to be equivalent to whole stele apoplasm, we only account for the cumulated resistance of cortex and epidermis cell wals.", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    kr_symplasmic_water_phloem: float = declare(default=1., unit="m3.s-1.Pa-1", unit_comment="", description="Effective Symplasmic water conductance of all cell layer contribution, including transmembrane and plasmodesmata resistance", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    xylem_differentiation_factor: float = declare(default=1., unit="adim", unit_comment="of vessel membrane", description="",
                                            min_value="", max_value="", value_comment="", references="",  DOI="", 
                                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")

    # FROM GROWTH MODEL
    length: float = declare(default=0, unit="m", unit_comment="of root segment", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    radius: float = declare(default=0, unit="m", unit_comment="of root segment", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    living_struct_mass: float = declare(default=0, unit="g", unit_comment="of dry weight", description="", 
                                 min_value="", max_value="", value_comment="", references="", DOI="",
                                 variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    type: str = declare(default="Normal_root_after_emergence", unit="", unit_comment="", description="Example segment type provided by root growth model", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="", edit_by="user")

    # FROM SHOOT MODEL
    water_root_shoot_xylem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="Transpiration related flux at collar", 
                                            min_value="", max_value="", value_comment="", references="", DOI="",
                                            variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    xylem_pressure_collar: float = declare(default=-0.5e6, unit="Pa", unit_comment="", description="Xylem water pressure at collar", 
                                            min_value="", max_value="", value_comment="", references="For young seedlings, supposed quasi stable McGowan and Tzimas", DOI="",
                                            variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    phloem_pressure_collar: float = declare(default=2e6, unit="Pa", unit_comment="", description="Phloem water potential at collar", 
                                            min_value="", max_value="", value_comment="", references="Dinant et al. 2010 for Barley", DOI="",
                                            variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    
    # FROM METABOLIC MODELS
    Cv_solute_xylem: float = declare(default=0., unit="mol.m-3", unit_comment="of total solutes", description="Total solute concentration in xylem", 
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="input", by="model_soil", state_variable_type="", edit_by="user")
    C_solute_phloem: float = declare(default=1., unit="mol.m-3", unit_comment="of total solutes", description="Total solute concentration in phloem", 
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="input", by="model_soil", state_variable_type="", edit_by="user")

    # --- INITIALIZE MODEL STATE VARIABLES ---

    # LOCAL VARIABLES

    # Pools initial values
    xylem_water: float = declare(default=0, unit="m3", unit_comment="of water", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    phloem_water: float = declare(default=0, unit="m3", unit_comment="of water", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    xylem_pressure_in: float = declare(default=-0.01e6, unit="Pa", unit_comment="", description="apoplastic pressure in stele at rest, we want the -0.5e6 target to be emerging from water balance", 
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    xylem_pressure_out: float = declare(default=-0.01e6, unit="Pa", unit_comment="", description="apoplastic pressure in stele at rest, we want the -0.5e6 target to be emerging from water balance", 
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    phloem_pressure_in: float = declare(default=1e6, unit="Pa", unit_comment="", description="apoplastic pressure in stele at rest, we want the -0.5e6 target to be emerging from water balance", 
                                          min_value="", max_value="", value_comment="", references="Dinant et al. 2010", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    phloem_pressure_out: float = declare(default=1e6, unit="Pa", unit_comment="", description="apoplastic pressure in stele at rest, we want the -0.5e6 target to be emerging from water balance", 
                                          min_value="", max_value="", value_comment="", references="Dinant et al. 2010", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    
    # Conductance values
    kr_xylem: float = declare(default=0, unit="m3.Pa-1.s-1", unit_comment="", description="radial root segment conductance", 
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    K_xylem: float = declare(default=0, unit="m3.Pa-1.s-1", unit_comment="", description="axial root segment conductance", 
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    K_phloem: float = declare(default=0, unit="m3.Pa-1.s-1", unit_comment="", description="axial root segment conductance", 
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    Keq: float = declare(default=0, unit="m3.Pa-1.s-1", unit_comment="", description="Equivalent conductance of the current root segment considering its position in the root system", 
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")

    # Water properties
    # sap_viscosity: float = declare(default=1.003e-3, unit="Pa.s", unit_comment="", description="Viscosity at 20°C", 
    #                                min_value="0.535e-3", max_value="1.753e-3", value_comment="", references="", DOI="",
    #                                variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")

    # Water transport processes
    radial_import_water_xylem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="", 
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    radial_import_water_xylem_apoplastic: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="Water flow through the apoplastic pathway, computed for radial advection", 
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    axial_export_water_up_xylem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="",
                                           min_value="", max_value="", value_comment="", references="", DOI="",
                                           variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    axial_import_water_down_xylem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="",
                                             min_value="", max_value="", value_comment="", references="", DOI="",
                                             variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    
    radial_import_water_phloem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="radial water exchange between xylem and phloem, mostly osmotic driven", 
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    axial_export_water_up_phloem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="",
                                           min_value="", max_value="", value_comment="", references="", DOI="",
                                           variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    axial_import_water_down_phloem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="",
                                             min_value="", max_value="", value_comment="", references="", DOI="",
                                             variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")

    # --- INITIALIZES MODEL PARAMETERS ---

    collar_flux_provided: bool = declare(default=False, unit="adim", unit_comment="", description="Option if collar flux is provided by input data", 
                                   min_value="", max_value="", value_comment="", references="", DOI="",
                                   variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    reflection_xylem: float = declare(default=0.85, unit="adim", unit_comment="", description="Reflection coefficient for soil-xylem radial water flux", 
                                   min_value="", max_value="", value_comment="", references="Miller, 1985a; Bauget et al., 2023", DOI="",
                                   variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    reflection_phloem: float = declare(default=0.85, unit="adim", unit_comment="", description="Reflection coefficient for phloem-xylem radial water flux", 
                                   min_value="", max_value="", value_comment="taken same as xylem", references="Miller, 1985a; Bauget et al., 2023", DOI="",
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
    def _K_xylem(self, soil_temperature, length, xylem_vessel_radii, xylem_differentiation_factor):
        """
        We assume xylem sap viscosity to be the same as that of water and use Andrade model to predict sap viscosity
        """
        A = 1.856e-11 * 1e-3 # Pa.s Viswanath & Natarajan (1989)
        B = 4209 # K Viswanath & Natarajan (1989)
        C = 0.04527 # K-1 Viswanath & Natarajan (1989)
        D = -3.376e-5 # K-2 Viswanath & Natarajan (1989)
        soil_temperature_Kelvin = soil_temperature + 273.15
        sap_viscosity = A * np.exp( (B / soil_temperature_Kelvin) + (C * soil_temperature_Kelvin) + D * (soil_temperature_Kelvin ** 2)) # Andrade 1930 polynomial extension by Viswanath & Natarajan (1989)
        return sum((np.pi * (vessel_radius ** 4) / (8 * sap_viscosity * length)) for vessel_radius in xylem_vessel_radii) * xylem_differentiation_factor
    
    @potential
    @rate
    def _K_phloem(self, C_solute_phloem, living_struct_mass, phloem_volume, soil_temperature, length, phloem_vessel_radii):
        """
        Haggen-Poiseuille model
        """
        solute_molar_volume = 160.35 * 1e-6 # m3.mol-1
        solute_volumetric_fraction = min(0.5, C_solute_phloem * living_struct_mass * solute_molar_volume / phloem_volume)
        # print("frac", solute_volumetric_fraction) # TODO: should not be constrained but here absurd values
        sap_viscosity = self.phloem_sap_viscosity(solute_volumetric_fraction, soil_temperature + 273.15)
        return sum((np.pi * (vessel_radius ** 4) / (8 * sap_viscosity * length)) for vessel_radius in phloem_vessel_radii)


    def phloem_sap_viscosity(self, solute_volumetric_fraction, soil_temperature_Kelvin):
        """
        Model from Telis et al. 2007, assuming sucrose properties for whole sap solutes
        """

        R = 8.314
        activation_energy_ref = 15080.24
        temperature_ref = 318.15 # 45°C
        # Fitted dependency of the ref viscosity for the abovedefined reference temperature (and converted to Pa.s-1)
        viscosity_ref_a = 9.6538
        viscosity_ref_b = 0.9706
        viscosity_ref_c = - 7.2891
        
        activation_energy = activation_energy_ref * (1 + (0.5 * solute_volumetric_fraction)) / (1 - solute_volumetric_fraction) # Telis et al. 2007
        viscosity_ref = np.exp((viscosity_ref_a * solute_volumetric_fraction**2) + viscosity_ref_b * solute_volumetric_fraction + viscosity_ref_c) # Pa.s-1 empirical
        return viscosity_ref * np.exp((activation_energy / R) * ((1/soil_temperature_Kelvin) - (1/ temperature_ref)))
    

    # @actual
    # @rate
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
        props = self.props

        # Select the base of the root
        root = next(g.component_roots_at_scale_iter(g.root, scale=1))

        # Equivalent conductance computation from tip to collar
        for v in post_order2(g, root):
            n = g.node(v)
            if n.living_struct_mass > 0.:
                if v == root:
                    children = self.collar_children
                else:
                    children = [child for child in g.children(v) if props["living_struct_mass"][child] > 0.]
                
                r = 1. / (n.kr_symplasmic_water_xylem + n.kr_apoplastic_water_xylem + sum(props["Keq"][cid] for cid in children))
                R = 1. / n.K_xylem
                n.Keq = 1. / (r + R)

        # Water flux and water potential computation from collar to tips
        for v in pre_order2(g, root):
            n = g.node(v)
            # Compute psi according to Millman theorem, then compute radial flux
            if n.living_struct_mass > 0:
                if v in self.collar_children:
                    parent = 1
                    brothers = self.collar_children
                else:
                    parent = g.parent(v)
                    brothers = [sibling for sibling in g.children_iter(parent) if props["living_struct_mass"][sibling] > 0.]
                p = g.node(parent)

                if v == root:
                    children = self.collar_children
                else:
                    children = [child for child in g.children_iter(v) if props["living_struct_mass"][child] > 0.]

                Keq_brothers = sum( props["Keq"][cid] for cid in brothers)
                Keq_children = sum( props["Keq"][cid] for cid in children)

                if parent is None:
                    n.xylem_pressure_out = props['xylem_pressure_collar'][1]

                    # If collar flux is provided by the shoot model
                    if self.collar_flux_provided:
                        n.axial_export_water_up_xylem = props['water_root_shoot_xylem'][1]
                    # Else we compute the flux according to the Haggen-Poiseuille conductance of
                    else:
                        n.axial_export_water_up_xylem = n.K_xylem * (n.xylem_pressure_in - n.xylem_pressure_out)

                else:
                    n.xylem_pressure_out = p.xylem_pressure_in
                    n.axial_export_water_up_xylem = (p.axial_export_water_up_xylem - p.radial_import_water_xylem) * ( n.Keq / Keq_brothers )
                
                k_radial_xylem = n.kr_symplasmic_water_xylem + n.kr_apoplastic_water_xylem
                n.kr_xylem = k_radial_xylem # TODO remove, only for visualization

                n.xylem_pressure_in = (n.K_xylem * n.xylem_pressure_out + n.soil_water_pressure * (k_radial_xylem + Keq_children)) / (k_radial_xylem + n.K_xylem + Keq_children)
                n.radial_import_water_xylem = (n.soil_water_pressure - n.xylem_pressure_in) * k_radial_xylem
                n.radial_import_water_xylem_apoplastic = (n.soil_water_pressure - n.xylem_pressure_in) * n.kr_apoplastic_water_xylem

                # Computed to avoid children iteration when needed by other modules
                if len(children) > 0:
                    n.axial_import_water_down_xylem = n.axial_export_water_up_xylem - n.radial_import_water_xylem
                else:
                    n.axial_import_water_down_xylem = 0


    @actual
    @rate
    def water_transport_munch(self):
        """the system of equation under matrix form is solved using a Newton-Raphson schemes, at each step a system J dY = -G
        is solved by LU decomposition.
        """
        g = self.g # To prevent repeated MTG lookups
        props = self.props
        struct_mass = g.property('struct_mass')

        # elt_number = len(g) - 1 #TODO: check
        local_vid = 1
        local_vids = {}
        for vid, value in struct_mass.items():
            if value > 0:
                local_vids[vid] = local_vid
                local_vid += 1

        elt_number = len(local_vids)
        minusG = np.zeros(2 * elt_number)
        
        # Select the base of the root
        root = next(g.component_roots_at_scale_iter(g.root, scale=1))

        ############
        # row and col indexes and non-zero Jacobian terms
        ############
        row = []
        col = []
        data = []

        for v in g.vertices_iter(scale = 1):

            n = g.node(v)
            
            if n.struct_mass > 0:
                # Volumic concentrations retreived there from inputs because metabolic only provides massic to be able to update on a growing arch 
                Cv_solute_xylem = n.C_solute_xylem * n.living_struct_mass / n.xylem_volume
                Cv_solute_phloem = n.C_solute_phloem * n.living_struct_mass / n.phloem_volume

                # Simulated separatly for apoplastic pathway decomposition, for phloem it is only symplastic so not differentiated
                kr_xylem = n.kr_symplasmic_water_xylem + n.kr_apoplastic_water_xylem
                n.kr_xylem = kr_xylem # TODO : Remove only for visualization
                kr_phloem = n.kr_symplasmic_water_phloem # Only a symplastic component

                if v == root:
                    children = self.collar_children
                    children_n = {cid: g.node(cid) for cid in children if struct_mass[cid] > 0}
                    p_parent_xylem = props['xylem_pressure_collar'][root]
                    p_parent_phloem = props['phloem_pressure_collar'][root]

                else:
                    children = g.children(v)
                    children_n = {cid: g.node(cid) for cid in children if struct_mass[cid] > 0}
                    if v in self.collar_children:
                        parent = root
                    else:
                        parent = g.parent(v)
                    p = g.node(parent)
                    p_parent_xylem = p.xylem_pressure_in
                    p_parent_phloem = p.phloem_pressure_in

                    # First block column
                    # dGp_xy_i/dP_xy_p
                    row.append(int(2 * local_vids[v] - 2))
                    col.append(int(2 * local_vids[parent] - 2))
                    data.append(- n.K_xylem)

                    # NOTE : Just kept for readability
                    # # dGp_ph_i/dP_xy_p
                    # row[nid] = int(2 * local_vids[v] - 1)
                    # col[nid] = int(2 * local_vids[parent] - 2)
                    # data[nid] = 0

                    # # Second block column
                    # # dGp_xy_i/dP_ph_p
                    # row[nid] = int(2 * local_vids[v] - 2)
                    # col[nid] = int(2 * local_vids[parent] - 1)
                    # data[nid] = 0

                    # dGp_ph_i/dP_ph_p
                    row.append(int(2 * local_vids[v] - 1))
                    col.append(int(2 * local_vids[parent] - 1))
                    data.append(- n.K_phloem)

                # First block column
                # dGp_xy_i/dP_xy_i
                row.append(int(2 * local_vids[v] - 2))
                col.append(int(2 * local_vids[v] - 2))
                data.append(n.K_xylem + sum([cn.K_xylem for cn in children_n.values()]) + kr_xylem + kr_phloem)

                # dGp_ph_i/dP_xy_i
                row.append(int(2 * local_vids[v] - 1))
                col.append(int(2 * local_vids[v] - 2))
                data.append(- kr_phloem)

                # Second block column
                # dGp_xy_i/dP_ph_i
                row.append(int(2 * local_vids[v] - 2))
                col.append(int(2 * local_vids[v] - 1))
                data.append(- kr_phloem)

                # dGp_ph_i/dP_ph_i
                row.append(int(2 * local_vids[v] - 1))
                col.append(int(2 * local_vids[v] - 1))
                data.append(n.K_phloem + sum([cn.K_phloem for cn in children_n.values()]) + kr_phloem)

                for cid, cn in children_n.items():
                    # First block column
                    # dGp_xy_i/dP_xy_j
                    row.append(int(2 * local_vids[v] - 2))
                    col.append(int(2 * local_vids[cid] - 2))
                    data.append(- cn.K_xylem)

                    # NOTE : Just kept for readability
                    # # dGp_ph_i/dP_xy_j
                    # row[nid] = int(2 * local_vids[v] - 1)
                    # col[nid] = int(2 * local_vids[cid] - 2)
                    # data[nid] = 0

                    # # Second block column
                    # # dGp_xy_i/dP_ph_j
                    # row[nid] = int(2 * local_vids[v] - 2)
                    # col[nid] = int(2 * local_vids[cid] - 1)
                    # data[nid] = 0

                    # dGp_ph_i/dP_ph_j
                    row.append(int(2 * local_vids[v] - 1))
                    col.append(int(2 * local_vids[cid] - 1))
                    data.append(- cn.K_phloem)

                # On growing architecture, pressure property has not been initialized on children here so we set it as that of the parent
                for cn in children_n.values():
                    # Only one check reveals an assignation need for both
                    if cn.xylem_pressure_in is None:
                        cn.xylem_pressure_in = n.xylem_pressure_in
                        cn.phloem_pressure_in = n.phloem_pressure_in

                # -Gp_xylem
                minusG[2 * local_vids[v] - 2] = -(n.K_xylem * (n.xylem_pressure_in - p_parent_xylem) 
                                    - sum([cn.K_xylem * (cn.xylem_pressure_in - n.xylem_pressure_in) for cn in children_n.values()]) 
                                    - kr_xylem * (n.soil_water_pressure - n.xylem_pressure_in - self.reflection_xylem * 8.31415 * (273.15 + n.soil_temperature) * (n.Cv_solute_soil - Cv_solute_xylem))
                                    - kr_phloem * (n.phloem_pressure_in - n.xylem_pressure_in - self.reflection_phloem * 8.31415 * (273.15 + n.soil_temperature) * (Cv_solute_phloem - Cv_solute_xylem)))
                # -Gp_phloem
                minusG[2 * local_vids[v] - 1] = -(n.K_phloem * (n.phloem_pressure_in - p_parent_phloem) 
                                    - sum([cn.K_phloem * (cn.phloem_pressure_in - n.phloem_pressure_in) for cn in children_n.values()]) 
                                    + kr_phloem * (n.phloem_pressure_in - n.xylem_pressure_in - self.reflection_phloem * 8.31415 * (273.15 + n.soil_temperature) * (Cv_solute_phloem - Cv_solute_xylem)))

        row = np.array(row, dtype=int)
        col = np.array(col, dtype=int)
        data = np.array(data, dtype=float)
        if debug: assert len(row) == len(row) == len(data)

        # NOTE for non standard cases (1 parent and more than 1 children): On main axis, no parent at collar and no children at root tip make 4 values fall out of the matrix
        # Then each branching (several on collar or simple lateral insertion), adds 2 terms being a supplementary children, but also substract 2 as it forms an apex.
        if debug: assert len(row) == 8 * elt_number - 4

        # Solving the system using sparse LU
        J = csc_matrix((data, (row, col)), shape = (2 * elt_number, 2 * elt_number))
        solve = linalg.splu(J)
        dY = solve.solve(minusG)

        # We apply results from collar to tips
        for v in pre_order2(g, root):
            n = g.node(v)

            if n.struct_mass > 0:

                # print(n.index(), n.xylem_pressure_in, dY[2 * local_vids[v] - 2], 2 * local_vids[v] - 2)
                if not np.isnan(dY[2 * local_vids[v] - 2]):
                    n.xylem_pressure_in = n.xylem_pressure_in + dY[2 * local_vids[v] - 2]
                else:
                    print("WARNING static xylem pressure")

                if not np.isnan(dY[2 * local_vids[v] - 1]):
                    n.phloem_pressure_in = n.phloem_pressure_in + dY[2 * local_vids[v] - 1]
                else:
                    print("WARNING static phloem pressure")

                if v == root:
                    n.xylem_pressure_out = props['xylem_pressure_collar'][root]
                    n.phloem_pressure_out = props['phloem_pressure_collar'][root]
                else:
                    if v in self.collar_children:
                        p = g.node(root)
                    else:
                        p = g.node(g.parent(v))
                    n.xylem_pressure_out = p.xylem_pressure_in
                    n.phloem_pressure_out = p.phloem_pressure_in

                n.axial_export_water_up_xylem = n.K_xylem * (n.xylem_pressure_in - n.xylem_pressure_out)
                n.axial_export_water_up_phloem = n.K_phloem * (n.phloem_pressure_in - n.phloem_pressure_out)

                Cv_solute_xylem = n.C_solute_xylem * n.living_struct_mass / n.xylem_volume
                Cv_solute_phloem = n.C_solute_phloem * n.living_struct_mass / n.phloem_volume

                n.radial_import_water_xylem = (n.kr_symplasmic_water_xylem + n.kr_apoplastic_water_xylem) * (n.soil_water_pressure - n.xylem_pressure_in - (self.reflection_xylem * 8.31415 * (273.15 + n.soil_temperature)) * (n.Cv_solute_soil - Cv_solute_xylem))
                n.radial_import_water_xylem_apoplastic = n.kr_apoplastic_water_xylem * (n.soil_water_pressure - n.xylem_pressure_in - (self.reflection_xylem * 8.31415 * (273.15 + n.soil_temperature)) * (n.Cv_solute_soil - Cv_solute_xylem))
                # Minus the orientation defined for G 
                # NOTE: Very important to keep this convention for vessel flux advection
                n.radial_import_water_phloem = - n.kr_symplasmic_water_phloem * (n.phloem_pressure_in - n.xylem_pressure_in - (self.reflection_phloem * 8.31415 * (273.15 + n.soil_temperature)) * (Cv_solute_phloem - Cv_solute_xylem))
                
                # Computed to avoid children iteration when needed by other modules
                n.axial_import_water_down_xylem = n.axial_export_water_up_xylem - n.radial_import_water_xylem + n.radial_import_water_phloem # That last one was reversed compared to Gminus so it enters phloem
                n.axial_import_water_down_phloem = n.axial_export_water_up_phloem - n.radial_import_water_phloem
                if debug: assert np.abs(n.axial_export_water_up_xylem + n.radial_import_water_phloem - n.axial_import_water_down_xylem - n.radial_import_water_xylem) < 1e-18
                if debug: assert np.abs(n.axial_export_water_up_phloem - n.axial_import_water_down_phloem - n.radial_import_water_phloem) < 1e-18

                if len(g.children(v)) == 0:
                    if debug: assert np.abs(n.axial_import_water_down_xylem) < 1e-18
                    if debug: assert np.abs(n.axial_import_water_down_phloem) < 1e-18

                # Usefull visual checks
                # print(n.index(), n.phloem_pressure_in, n.kr_symplasmic_water_phloem, n.axial_export_water_up_phloem, n.radial_import_water_phloem, n.axial_import_water_down_phloem, Cv_solute_phloem, Cv_solute_xylem)
                # print(n.index(), n.xylem_pressure_in, n.kr_symplasmic_water_xylem, n.soil_water_pressure, n.axial_export_water_up_xylem, n.radial_import_water_xylem, n.axial_import_water_down_xylem)


    @state
    def _xylem_water(self, xylem_volume):
        # return xylem_volume * 1e6 / 18
        return xylem_volume
    
    @state
    def _phloem_water(self, phloem_volume):
        # return xylem_volume * 1e6 / 18
        return phloem_volume
