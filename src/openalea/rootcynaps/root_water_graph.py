import numpy as np
import time
from openalea.mtg.traversal import pre_order2
from dataclasses import dataclass
from openalea.mtg.traversal import post_order2, pre_order2

from openalea.metafspm.component import FunctionalComponent, declare
from openalea.metafspm.component_factory import *
from openalea.metafspm.mpg import MPG
from openalea.metafspm.graph_system import GraphView
from openalea.metafspm.graph_system_decorators import graph_system, node_balance, boundary_condition, graph_jacobian, graph_output

from scipy.sparse import csc_matrix, diags, linalg


debug = True

@dataclass
class RootWaterModel(FunctionalComponent):


    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

    # FROM SOIL MODEL
    soil_water_pressure: float = declare(default=0., unit="Pa", unit_comment="of water", description="",
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="input", by="model_soil", state_variable_type="", edit_by="user")
    soil_temperature: float = declare(default=7.8, unit="°C", unit_comment="", description="soil temperature in contact with roots",
                                        min_value="", max_value="", value_comment="Derived from Swinnen et al. 1994 C inputs, estimated from a labelling experiment starting 3rd of March, with average temperature at 7.8 °C", references="Swinnen et al. 1994", DOI="",
                                        variable_type="input", by="model_temperature", state_variable_type="", edit_by="user")
    Cv_solutes_soil: float = declare(default=0., unit="mol.m-3", unit_comment="of total solutes", description="Total solute concentration in soil",
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
    xylem_differentiation_factor: float = declare(default=3., unit="adim", unit_comment="of vessel membrane", description="",
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
    water_root_shoot_xylem: float = declare(default=None, unit="m3.s-1", unit_comment="of water", description="Transpiration related flux at collar",
                                            min_value="", max_value="", value_comment="", references="", DOI="",
                                            variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    xylem_pressure_collar: float = declare(default=-0.5e6, unit="Pa", unit_comment="", description="Xylem water pressure at collar",
                                            min_value="", max_value="", value_comment="", references="For young seedlings, supposed quasi stable McGowan and Tzimas", DOI="",
                                            variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    phloem_pressure_collar: float = declare(default=2e6, unit="Pa", unit_comment="", description="Phloem water potential at collar",
                                            min_value="", max_value="", value_comment="", references="Dinant et al. 2010 for Barley", DOI="",
                                            variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    Cv_sucrose_phloem_collar: float = declare(default=950, unit="mol.m-3", unit_comment="", description="Sucrose volumic concentration in phloem at collar point", 
                                       min_value=0, max_value=1200, value_comment="", references="Winter et al. 1992", DOI="",
                                        variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    sucrose_root_to_shoot_phloem: float = declare(default=None, unit="mol.s-1", unit_comment="", description="Sucrose input rate in phloem at collar point", 
                                       min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")

    # FROM METABOLIC MODELS
    C_solutes_xylem: float = declare(default=0., unit="mol.m-3", unit_comment="of total solutes", description="Total solute concentration in xylem",
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="input", by="model_soil", state_variable_type="", edit_by="user")
    C_solutes_phloem: float = declare(default=1., unit="mol.m-3", unit_comment="of total solutes", description="Total solute concentration in phloem",
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
    xylem_pressure: float = declare(default=-0.01e6*5, unit="Pa", unit_comment="", description="apoplastic pressure in stele at rest, we want the -0.5e6 target to be emerging from water balance",
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    phloem_pressure: float = declare(default=1e6, unit="Pa", unit_comment="", description="apoplastic pressure in stele at rest, we want the -0.5e6 target to be emerging from water balance",
                                          min_value="", max_value="", value_comment="", references="Dinant et al. 2010", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")

    # Conductance values
    kr_xylem: float = declare(default=0, unit="m3.Pa-1.s-1", unit_comment="", description="radial root segment conductance",
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    K_xylem: float = declare(default=0, unit="m3.Pa-1.s-1", unit_comment="", description="axial root segment conductance",
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user",
                                          location="edge")
    K_phloem: float = declare(default=0, unit="m3.Pa-1.s-1", unit_comment="", description="axial root segment conductance",
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user",
                                          location="edge")

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
    axial_water_flow_xylem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="",
                                           min_value="", max_value="", value_comment="", references="", DOI="",
                                           variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")

    radial_import_water_phloem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="radial water exchange between xylem and phloem, mostly osmotic driven",
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    axial_water_flow_phloem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="",
                                           min_value="", max_value="", value_comment="", references="", DOI="",
                                           variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")

    # Graph-system fields (used by @graph_system _transport_solve)
    osmotic_xylem_term: float = declare(default=0., unit="Pa", unit_comment="", description="Pre-computed osmotic correction for soil-xylem radial exchange: reflection_xylem * RT * (Cv_soil - Cv_xylem)",
                                        min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user",
                                        location="node")
    osmotic_phloem_term: float = declare(default=0., unit="Pa", unit_comment="", description="Pre-computed osmotic correction for phloem-xylem radial exchange: reflection_phloem * RT * (Cv_phloem - Cv_xylem)",
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user",
                                         location="node")
    is_collar: float = declare(default=0., unit="adim", unit_comment="", description="1.0 at the collar node (vid=1), 0.0 elsewhere; populated by _update_graph_view for use as a types filter",
                               min_value="", max_value="", value_comment="", references="", DOI="",
                               variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user",
                               location="node")

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

    # Helpers to keep labels intergers
    label_Segment: int = declare(default=1, unit="adim", unit_comment="", description="label utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    label_Apex: int = declare(default=2, unit="adim", unit_comment="", description="label utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")


    # Helpers to keep types intergers
    type_Base_of_the_root_system: int = declare(default=1, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Support_for_seminal_root: int = declare(default=2, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Seminal_root_before_emergence: int = declare(default=3, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Support_for_adventitious_root: int = declare(default=4, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Adventitious_root_before_emergence: int = declare(default=5, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Normal_root_before_emergence: int = declare(default=6, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Normal_root_after_emergence: int = declare(default=7, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Stopped: int = declare(default=8, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Just_stopped: int = declare(default=9, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Dead: int = declare(default=10, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Just_dead: int = declare(default=11, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Root_nodule: int = declare(default=12, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")


    def __init__(self, g, time_step, **scenario):
        """
        Description :
        This root water model discretized at root segment's scale intends to account for heterogeneous axial and radial water flows observed in the roots (Bauget et al. 2022).

        Hypothesis :
        Accounting for heterogeneous water flows would improbe the overall nutrient balance for root hydromineral uptake.
        """
        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)

        self.g = g
        self.props = self.g.properties()
        self.time_step = time_step
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step, data=self.props)
        self.vertices = self.g.vertices(scale=self.g.max_scale())

        self.link_self_to_mtg()


    def post_coupling_init(self):
        self.pull_available_inputs()


        # SPECIFIC HERE, Select real children for collar element (vid == 1).
        # This is mandatory for correct collar-to-tip Hagen-Poiseuille flow partitioning.
        self.collar_children, self.collar_skip = [], []
        for vid in self.vertices:
            children = self.g.children(vid)
            if self.type[vid] in (self.type_Support_for_seminal_root, self.type_Support_for_adventitious_root) and children:
                self.collar_skip += [vid]
                self.collar_children += [k for k in children if self.type[k] not in (self.type_Support_for_seminal_root, self.type_Support_for_adventitious_root)]

        self._rebuild_graph_view()

    @stepinit
    def _update_graph_view(self):
        """Rebuild _graph_view each timestep to track growing root architecture."""
        self._rebuild_graph_view()

    def _rebuild_graph_view(self):
        """
        Build (or rebuild) self._graph_view from the current focus_elements set.

        Called once from post_coupling_init (after pull_available_inputs) and
        each timestep from the @stepinit _update_graph_view method.

        Side-effects:
          - self.props["is_collar"] is updated: 1.0 at the collar node, 0.0 elsewhere.
          - self._graph_view is replaced with the new GraphView.
        """
        props = self.props

        # Focus VIDs: living active segments
        focus_vids_list = [int(v) for v in props["focus_elements"]]

        # Skip structural connector segments (Support_for_seminal/adventitious_root)
        skip_set = set(self.collar_skip)

        mpg = MPG.from_mtg(self.g)
        is_collar_dict = mpg.populate_node_edge_scales(
            focus_vids_list,
            skip_predicate=lambda v: v in skip_set,
        )

        # Write is_collar into props (1.0 = collar, 0.0 = interior node)
        is_collar_prop = props["is_collar"]
        for vid, val in is_collar_dict.items():
            is_collar_prop[vid] = 1.0 if val else 0.0

        # Node IDs = all non-skipped focus nodes; edge IDs = non-collar nodes (children)
        node_vids = np.array(sorted(is_collar_dict.keys()), dtype=np.int64)
        edge_vids = np.array(
            sorted(v for v, c in is_collar_dict.items() if not c), dtype=np.int64
        )

        self._graph_view = GraphView.from_mtg_subset(
            mpg,
            node_scale=MPG.scales["node"],
            node_ids=node_vids,
            edge_scale=MPG.scales["edge"],
            edge_ids=edge_vids,
        )

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
        # print(sap_viscosity)
        return np.sum([(np.pi * (vessel_radius ** 4) / (8 * sap_viscosity * length)) for vessel_radius in xylem_vessel_radii]) * xylem_differentiation_factor

    @potential
    @rate
    def _K_phloem(self, C_solutes_phloem, living_struct_mass, phloem_volume, soil_temperature, length, phloem_vessel_radii):
        """
        Haggen-Poiseuille model
        """
        solute_molar_volume = 160.35 * 1e-6 # m3.mol-1
        # solute_molar_volume = 100 * 1e-6 # m3.mol-1
        solute_volumetric_fraction = np.maximum(0., np.minimum(0.1, C_solutes_phloem * living_struct_mass * solute_molar_volume / phloem_volume))
        # print("fraction", solute_volumetric_fraction)
        # print("frac",  C_solutes_phloem * living_struct_mass * solute_molar_volume / phloem_volume) # TODO: should not be constrained but here absurd values
        sap_viscosity = self.phloem_sap_viscosity(solute_volumetric_fraction, soil_temperature + 273.15)
        # print(sap_viscosity)
        return np.sum([(np.pi * (vessel_radius ** 4) / (8 * sap_viscosity * length)) for vessel_radius in phloem_vessel_radii])


    @rate
    def _osmotic_xylem_term(self, soil_temperature, Cv_solutes_soil, C_solutes_xylem, living_struct_mass, xylem_volume):
        """reflection_xylem * RT * (Cv_soil - Cv_xylem), pre-computed per node before the graph solve."""
        RT = 8.31415 * (273.15 + soil_temperature)
        Cv_xylem = C_solutes_xylem * living_struct_mass / xylem_volume if xylem_volume > 0. else 0.
        return self.reflection_xylem * RT * (Cv_solutes_soil - Cv_xylem)

    @rate
    def _osmotic_phloem_term(self, soil_temperature, C_solutes_xylem, C_solutes_phloem, living_struct_mass, xylem_volume, phloem_volume):
        """reflection_phloem * RT * (Cv_phloem - Cv_xylem), pre-computed per node before the graph solve."""
        RT = 8.31415 * (273.15 + soil_temperature)
        Cv_xylem = C_solutes_xylem * living_struct_mass / xylem_volume if xylem_volume > 0. else 0.
        Cv_phloem = C_solutes_phloem * living_struct_mass / phloem_volume if phloem_volume > 0. else 0.
        return self.reflection_phloem * RT * (Cv_phloem - Cv_xylem)

    @graph_system(
        node_unknowns=["xylem_pressure", "phloem_pressure"],
        edge_unknowns=[],
        method="newton",
        max_iter=2,
        tol=1e-8,
        schedule_as="axial",
    )
    class _transport_solve:

        # ── Xylem residual (all nodes; Dirichlet BC overwrites collar row) ────

        @node_balance(field="xylem_pressure")
        def _xylem_balance_residual(self, xylem_pressure, phloem_pressure, K_xylem,
                                 kr_symplasmic_water_xylem, kr_apoplastic_water_xylem,
                                 kr_symplasmic_water_phloem,
                                 soil_water_pressure, osmotic_xylem_term, osmotic_phloem_term):
            incidence = self._graph_view.incidence
            laplacian_xylem = incidence @ diags(K_xylem) @ incidence.T
            kr_xylem_total = kr_symplasmic_water_xylem + kr_apoplastic_water_xylem
            return (np.asarray(laplacian_xylem @ xylem_pressure).reshape(-1)
                    - kr_xylem_total * (soil_water_pressure - xylem_pressure - osmotic_xylem_term)
                    - kr_symplasmic_water_phloem * (phloem_pressure - xylem_pressure - osmotic_phloem_term))

        # ── Xylem collar BC — two mutually exclusive options ─────────────────────
        # OPTION A (active): Dirichlet — prescribes collar pressure from shoot model.
        #   Use when water_root_shoot_xylem is None (no transpiration flux provided).
        @boundary_condition("node", "dirichlet", field="xylem_pressure", types={"is_collar": [1.0]}, explicit=True)
        def _xylem_collar_dirichlet(self):
            return np.array([self.props["xylem_pressure_collar"][1]])

        # OPTION B (inactive): Neumann — injects transpiration as a flux source at the collar,
        #   letting Newton solve the collar pressure self-consistently from the full balance.
        #   Use when water_root_shoot_xylem is provided. The Jacobian adapts automatically.
        #   To activate: uncomment this block and comment out OPTION A above.
        # @boundary_condition("node", "neumann", field="xylem_pressure", types={"is_collar": [1.0]})
        # def _xylem_collar_neumann(self):
        #     return np.array([self.props["water_root_shoot_xylem"][1]])

        # ── Phloem residual (all nodes; Neumann BC adds flux at collar row) ──

        @node_balance(field="phloem_pressure"):
        def _phloem_balance_residual(self, xylem_pressure, phloem_pressure, K_phloem,
                                  kr_symplasmic_water_phloem, osmotic_phloem_term):
            incidence = self._graph_view.incidence
            laplacian_phloem = incidence @ diags(K_phloem) @ incidence.T
            return (np.asarray(laplacian_phloem @ phloem_pressure).reshape(-1)
                    + kr_symplasmic_water_phloem * (phloem_pressure - xylem_pressure - osmotic_phloem_term))

        # ── Phloem collar BC — two mutually exclusive options ────────────────────
        # OPTION A (inactive): Dirichlet — prescribes collar pressure from shoot model.
        #   Use when sucrose_root_to_shoot_phloem is None (no sucrose flux provided).
        #   To activate: uncomment this block and comment out OPTION B below.
        # @boundary_condition("node", "dirichlet", field="phloem_pressure", types={"is_collar": [1.0]}, explicit=True)
        # def _phloem_collar_dirichlet(self):
        #     return np.array([self.props["phloem_pressure_collar"][1]])

        # OPTION B (active): Neumann — derives water flux from sucrose flux at collar,
        #   letting Newton solve collar pressure self-consistently from the full balance.
        #   Use when sucrose_root_to_shoot_phloem is provided. The Jacobian adapts automatically.
        #   Negative sucrose flux uses local collar concentration; positive uses system-wide mean.
        @boundary_condition("node", "neumann", field="phloem_pressure", types={"is_collar": [1.0]})
        def _phloem_collar_neumann(self):
            sucrose_flux = self.props["sucrose_root_to_shoot_phloem"][1]
            if sucrose_flux < 0.:
                phloem_sucrose_concentration = self.props["Cv_sucrose_phloem_collar"][1]
            else:
                phloem_sucrose_concentration = self.props["total_sucrose_phloem"][1] / sum(self.props["phloem_volume"].values())
            water_flux_phloem_collar = sucrose_flux / phloem_sucrose_concentration
            return np.array([-water_flux_phloem_collar])

        # ── Analytic Jacobian (2n × 2n) ───────────────────────────────────────

        @graph_jacobian
        def _analytic_jacobian(self, xylem_pressure, phloem_pressure,
                                K_xylem, K_phloem,
                                kr_symplasmic_water_xylem, kr_apoplastic_water_xylem,
                                kr_symplasmic_water_phloem, is_collar,
                                _node_fields):
            n_nodes = self._graph_view.n_nodes
            incidence = self._graph_view.incidence
            laplacian_xylem  = (incidence @ diags(K_xylem)  @ incidence.T).toarray()
            laplacian_phloem = (incidence @ diags(K_phloem) @ incidence.T).toarray()
            kr_xylem_total = kr_symplasmic_water_xylem + kr_apoplastic_water_xylem

            # Xylem collar: Dirichlet (OPTION A) → identity row; Neumann (OPTION B) → full bulk row.
            transpiration_flow = self.props["water_root_shoot_xylem"][1]
            dirichlet_collar_xylem = is_collar if (transpiration_flow is None or np.isnan(transpiration_flow)) else np.zeros(n_nodes)
            bulk_mask_xylem = (1.0 - dirichlet_collar_xylem)[:, None]

            # Phloem collar: Dirichlet (OPTION A) → identity row; Neumann (OPTION B) → full bulk row.
            sucrose_flux = self.props["sucrose_root_to_shoot_phloem"][1]
            dirichlet_collar_phloem = is_collar if (sucrose_flux is None or np.isnan(sucrose_flux)) else np.zeros(n_nodes)
            bulk_mask_phloem = (1.0 - dirichlet_collar_phloem)[:, None]

            xylem_block_offset  = _node_fields.index("xylem_pressure")  * n_nodes
            phloem_block_offset = _node_fields.index("phloem_pressure") * n_nodes

            jacobian = np.zeros((2 * n_nodes, 2 * n_nodes))
            jacobian[xylem_block_offset:xylem_block_offset+n_nodes,
                     xylem_block_offset:xylem_block_offset+n_nodes] = (
                (laplacian_xylem + np.diag(kr_xylem_total + kr_symplasmic_water_phloem)) * bulk_mask_xylem
                + np.diag(dirichlet_collar_xylem))
            jacobian[xylem_block_offset:xylem_block_offset+n_nodes,
                     phloem_block_offset:phloem_block_offset+n_nodes] = (
                -np.diag(kr_symplasmic_water_phloem) * bulk_mask_xylem)
            jacobian[phloem_block_offset:phloem_block_offset+n_nodes,
                     xylem_block_offset:xylem_block_offset+n_nodes] = (
                -np.diag(kr_symplasmic_water_phloem) * bulk_mask_phloem)
            jacobian[phloem_block_offset:phloem_block_offset+n_nodes,
                     phloem_block_offset:phloem_block_offset+n_nodes] = (
                (laplacian_phloem + np.diag(kr_symplasmic_water_phloem)) * bulk_mask_phloem
                + np.diag(dirichlet_collar_phloem))
            return jacobian

        # ── Post-solve outputs ────────────────────────────────────────────────

        @graph_output("axial_water_flow_xylem")
        def _axial_export_xylem(self, xylem_pressure, K_xylem):
            graph_view = self._graph_view
            axial_pressure_drop = np.asarray(graph_view.incidence.T @ xylem_pressure).reshape(-1)
            axial_flow = np.zeros(graph_view.n_nodes)
            axial_flow[graph_view.head] = -K_xylem * axial_pressure_drop
            return axial_flow

        @graph_output("axial_water_flow_phloem")
        def _axial_export_phloem(self, phloem_pressure, K_phloem):
            graph_view = self._graph_view
            axial_pressure_drop = np.asarray(graph_view.incidence.T @ phloem_pressure).reshape(-1)
            axial_flow = np.zeros(graph_view.n_nodes)
            axial_flow[graph_view.head] = -K_phloem * axial_pressure_drop
            return axial_flow

        @graph_output("radial_import_water_xylem")
        def _radial_import_xylem(self, xylem_pressure,
                                  kr_symplasmic_water_xylem, kr_apoplastic_water_xylem,
                                  soil_water_pressure, osmotic_xylem_term):
            return (kr_symplasmic_water_xylem + kr_apoplastic_water_xylem) * (
                soil_water_pressure - xylem_pressure - osmotic_xylem_term
            )

        @graph_output("radial_import_water_xylem_apoplastic")
        def _radial_import_xylem_apo(self, xylem_pressure,
                                      kr_apoplastic_water_xylem,
                                      soil_water_pressure, osmotic_xylem_term):
            return kr_apoplastic_water_xylem * (
                soil_water_pressure - xylem_pressure - osmotic_xylem_term
            )

        @graph_output("radial_import_water_phloem")
        def _radial_import_phloem(self, xylem_pressure, phloem_pressure,
                                   kr_symplasmic_water_phloem, osmotic_phloem_term):
            return -kr_symplasmic_water_phloem * (
                phloem_pressure - xylem_pressure - osmotic_phloem_term
            )

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


    @state
    def _xylem_water(self, xylem_volume):
        # return xylem_volume * 1e6 / 18
        return xylem_volume


    @state
    def _phloem_water(self, phloem_volume):
        # return xylem_volume * 1e6 / 18
        return phloem_volume
