"""
root_cynaps.nitrogen
_________________
This is the main nitrogen cycle module for roots

Documentation and features
__________________________

Main functions
______________
Classes' names represent accounted hypothesis in the progressive development of the model.
Methods' names are systematic through all class for ease of use :
"""

# Imports
import numpy as np
from dataclasses import dataclass
import time

from openalea.metafspm.component import Model, declare
from openalea.metafspm.component_factory import *

from scipy.sparse import csc_matrix, identity, linalg, diags
from scipy.optimize import lsq_linear
from sksundae.ida import IDA
from scipy.integrate import solve_ivp

debug = True

@dataclass
class RootNitrogenModel(Model):
    """
    Root nitrogen balance model of Root_CyNAPS
    """


    # --- @note INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

    # FROM CARBON MODEL
    C_hexose_root: float = declare(default=1e-4, unit="mol.g-1", unit_comment="of labile hexose", description="Hexose concentration in root",
                                  min_value="", max_value="", value_comment="", references="", DOI="",
                                   variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    

    # FROM SOIL MODEL
    soil_Nm: float = declare(default=2.2, unit="mol.m-3", unit_comment="of nitrates", description="", 
                            min_value="", max_value="", value_comment="", references="Fischer 1966", DOI="", 
                            variable_type="input", by="model_soil", state_variable_type="", edit_by="user")
    soil_AA: float = declare(default=8.2e-3, unit="mol.m-3", unit_comment="of amino acids", description="",
                            min_value="", max_value="", value_comment="Fischer et al 2007, water leaching estimation", references="", DOI="", 
                            variable_type="input", by="model_soil", state_variable_type="", edit_by="user")
    struct_mass_fungus: float = declare(default=1e-3, unit="g", unit_comment="of hyphal structural mass", description="",
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="input", by="model_soil", state_variable_type="", edit_by="user")
    Nm_fungus: float = declare(default=1e-4, unit="mol.g-1", unit_comment="mol of inorganic nitrogen per g or hyphal structural mass", description="",
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="input", by="model_soil", state_variable_type="", edit_by="user")
    soil_temperature: float = declare(default=20, unit="°C", unit_comment="", description="soil temperature in contact with roots",
                            value_comment="", references="", DOI="",
                            min_value="", max_value="", variable_type="input", by="model_temperature", state_variable_type="", edit_by="user")

    # FROM ANATOMY MODEL
    root_exchange_surface: float = declare(default=0, unit="m2", unit_comment="of cell membrane", description="",
                                           min_value="", max_value="", value_comment="", references="",  DOI="", 
                                           variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    xylem_exchange_surface: float = declare(default=0, unit="m2", unit_comment="of cell membrane", description="Surface of xylem aproximated as the one of the stele", 
                                            min_value="", max_value="", value_comment="", references="",  DOI="", 
                                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    phloem_exchange_surface: float = declare(default=0, unit="m2", unit_comment="of vessel membrane", description="",
                                            min_value="", max_value="", value_comment="", references="",  DOI="", 
                                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    xylem_differentiation_factor: float = declare(default=1., unit="adim", unit_comment="of vessel membrane", description="",
                                            min_value="", max_value="", value_comment="", references="",  DOI="", 
                                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    endodermis_conductance_factor: float = declare(default=1., unit="adim", unit_comment="of vessel membrane", description="",
                                            min_value="", max_value="", value_comment="", references="",  DOI="", 
                                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    symplasmic_volume: float = declare(default=1e-9, unit="m3", unit_comment="", description="symplasmic volume for water content of root elements", 
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_anatomy", state_variable_type="extensive", edit_by="user")
    xylem_volume: float = declare(default=0., unit="m3", unit_comment="", description="", 
                                  min_value="", max_value="", value_comment="", references="", DOI="",
                                  variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    phloem_volume: float = declare(default=0., unit="m3", unit_comment="", description="", 
                                  min_value="", max_value="", value_comment="", references="", DOI="",
                                  variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    total_phloem_volume: float = declare(default=0., unit="m3", unit_comment="", description="", 
                                  min_value="", max_value="", value_comment="", references="", DOI="",
                                  variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
                                            

    # FROM WATER BALANCE MODEL
    xylem_water: float =                declare(default=0, unit="m3", unit_comment="of water", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_water", state_variable_type="", edit_by="user")
    phloem_water: float =                declare(default=0, unit="m3", unit_comment="of water", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_water", state_variable_type="", edit_by="user")
    radial_import_water_xylem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="", 
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="input", by="model_water", state_variable_type="", edit_by="user")
    radial_import_water_xylem_apoplastic: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="", 
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="input", by="model_water", state_variable_type="", edit_by="user")
    axial_export_water_up_xylem: float =      declare(default=0, unit="m3.s-1", unit_comment="of water", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_water", state_variable_type="", edit_by="user")
    axial_import_water_down_xylem: float =    declare(default=0, unit="m3.s-1", unit_comment="of water", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_water", state_variable_type="", edit_by="user")
    radial_import_water_phloem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="", 
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="input", by="model_water", state_variable_type="", edit_by="user")
    axial_export_water_up_phloem: float =      declare(default=0, unit="m3.s-1", unit_comment="of water", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_water", state_variable_type="", edit_by="user")
    axial_import_water_down_phloem: float =    declare(default=0, unit="m3.s-1", unit_comment="of water", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_water", state_variable_type="", edit_by="user")

    # FROM GROWTH MODEL
    type: int = declare(default=1, unit="", unit_comment="", description="Example segment type provided by root growth model", 
                       min_value="", max_value="", value_comment="", references="", DOI="",
                        variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    length: float =                     declare(default=0, unit="m", unit_comment="of root segment", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    radius: float =                     declare(default=0, unit="m", unit_comment="of root segment", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    struct_mass: float =                declare(default=0, unit="g", unit_comment="of dry weight", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    living_struct_mass: float =                declare(default=0, unit="g", unit_comment="of dry weight", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    total_living_struct_mass: float =          declare(default=0, unit="g", unit_comment="of dry weight", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    hexose_consumption_by_growth: float = declare(default=0., unit="mol.s-1", unit_comment="", description="Hexose consumption rate by growth is coupled to a root growth model", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    thermal_time_since_emergence: float = declare(default=0, unit="°C", unit_comment="", description="", 
                                                  min_value="", max_value="", value_comment="", references="", DOI="",
                                                  variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    distance_from_tip: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example distance from tip", 
                                      min_value="", max_value="", value_comment="", references="", DOI="",
                                       variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    vertex_index: int = declare(default=1, unit="adim", unit_comment="", description="Unique vertex identifier stored for ease of value access", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="extensive", edit_by="user")
    hexose_diffusion_from_phloem_rate: float = declare(default=0., unit="mol.s-1", unit_comment="", description="Hexose unloading rate from phloem", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")

    # FROM SHOOT MODEL
    Cv_Nm_xylem_collar: float = declare(default=1, unit="mol.m-3", unit_comment="", description="Sucrose input rate in phloem at collar point", 
                                       min_value="", max_value="", value_comment="range approximation", references="", DOI="",
                                        variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    Cv_AA_xylem_collar: float = declare(default=0.1, unit="mol.m-3", unit_comment="", description="Sucrose input rate in phloem at collar point", 
                                       min_value="", max_value="", value_comment="range approximation", references="", DOI="",
                                        variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    Cv_AA_phloem_collar: float = declare(default=260*2, unit="mol.m-3", unit_comment="", description="Sucrose input rate in phloem at collar point", 
                                       min_value="", max_value="", value_comment="", references="Dinant et al. 2010", DOI="",
                                        variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    AA_input_rate_phloem: float = declare(default=None, unit="mol.s-1", unit_comment="", description="Amino acids input rate in phloem at root-shoot junction", 
                                       min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    AA_input_rate_xylem: float = declare(default=None, unit="mol.s-1", unit_comment="", description="Amino acids input rate in xylem at root-shoot junction", 
                                       min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    Nm_input_rate_xylem: float = declare(default=None, unit="mol.s-1", unit_comment="", description="Sucrose input rate in phloem at root-shoot junction", 
                                       min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    cytokinins_root_shoot_xylem: float = declare(default=0, unit="mol.h-1", unit_comment="of cytokinins", description="",
                                                 min_value="", max_value="", value_comment="", references="", DOI="",
                                                 variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")

    # --- @note INITIALIZE MODEL STATE VARIABLES --- 

    # LOCAL VARIABLES

    # Pools initial size
    Nm: float =                 declare(default=1e-4, unit="mol.g-1", unit_comment="of nitrates", description="",
                                        min_value=1e-6, max_value=1e-3, value_comment="", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="massic_concentration", edit_by="user")
    AA: float =                 declare(default=2.6e-5, unit="mol.g-1", unit_comment="of amino acids", description="",
                                        min_value=1e-5, max_value=1e-2, value_comment="increased from expected 2.6e-5 for initial stability", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="massic_concentration", edit_by="user")
    storage_protein: float =    declare(default=0., unit="mol.g-1", unit_comment="of storage proteins", description="", 
                                        min_value="", max_value="", value_comment="0 value for wheat", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="massic_concentration", edit_by="user")
    xylem_Nm: float =           declare(default=1e-4 / 100 / 10, unit="mol.g-1", unit_comment="of structural nitrates", description="", 
                                        min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="massic_concentration", edit_by="user")
    xylem_AA: float =           declare(default=1e-4 / 100 / 2, unit="mol.g-1", unit_comment="of amino acids", description="", 
                                        min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="massic_concentration", edit_by="user")
    phloem_AA: float =           declare(default=1e-4, unit="mol.g-1", unit_comment="of amino acids", description="", 
                                        min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="massic_concentration", edit_by="user")
    
    # Agregates for the water transport model 
    C_solutes_xylem: float =                 declare(default=0, unit="mol.m-3", unit_comment="of total solutes", description="Total solute concentration in xylem",
                                        min_value=1e-6, max_value=1e-3, value_comment="", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="massic_concentration", edit_by="user")
    C_solutes_phloem: float =                 declare(default=1e-3, unit="mol.g-1", unit_comment="of total solutes", description="Total solute concentration in phloem",
                                        min_value=1e-6, max_value=1e-3, value_comment="", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="massic_concentration", edit_by="user")
    
    # Transport processes
    import_Nm: float =                      declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                    min_value=1e-11, max_value=1e-9, value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    import_Nm_LATS: float =                      declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                    min_value=1e-11, max_value=1e-9, value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    import_AA: float =                      declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    export_Nm: float =                      declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    export_AA: float =                      declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    diffusion_Nm_soil: float =              declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    diffusion_Nm_xylem: float =             declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="",
                                                    min_value="", max_value="", value_comment="", references="", DOI="", 
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    apoplastic_Nm_soil_xylem: float =        declare(default=0., unit="mol.s-1", unit_comment="of nitrates", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    diffusion_AA_soil: float =              declare(default=0., unit="mol.s-1", unit_comment="of amino acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    diffusion_AA_phloem: float =            declare(default=0., unit="mol.s-1", unit_comment="of amino acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    unloading_AA_phloem: float =            declare(default=0., unit="mol.s-1", unit_comment="of amino acids", 
                                                    min_value="", max_value="", description="Active import of amino acids from phloem, in line with dual flow from rhizodep", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    apoplastic_AA_soil_xylem: float =        declare(default=0., unit="mol.s-1", unit_comment="of amino acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    
    # Metabolic processes
    AA_synthesis: float =                   declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    storage_synthesis: float =              declare(default=0., unit="mol.s-1", unit_comment="of storage", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    AA_catabolism: float =                  declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    storage_catabolism: float =             declare(default=0., unit="mol.s-1", unit_comment="of storage", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    # NOTE : temporary for root CyNAPS outputs when used alone, otherwise should just be an input of the model
    amino_acids_consumption_by_growth: float =             declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")

    # Axial transport processes
    
    displaced_Nm_in_xylem: float =                declare(default=0., unit="mol.time_step-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    displaced_Nm_out_xylem: float =               declare(default=0., unit="mol.time_step-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    Nm_differential_by_water_transport: float =    declare(default=0., unit="mol.time_step-1", unit_comment="of nitrates", description="", 
                                                    min_value=-1e9, max_value=1e9, value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    displaced_AA_in_xylem: float =                declare(default=0., unit="mol.time_step-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    displaced_AA_out_xylem: float =               declare(default=0., unit="mol.time_step-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    cumulated_radial_exchanges_Nm_xylem: float =  declare(default=0., unit="mol.time_step-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    cumulated_radial_exchanges_AA_xylem: float =  declare(default=0., unit="mol.time_step-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    displaced_AA_in_phloem: float =                declare(default=0., unit="mol.time_step-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    displaced_AA_out_phloem: float =               declare(default=0., unit="mol.time_step-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    cumulated_radial_exchanges_AA_phloem: float =  declare(default=0., unit="mol.time_step-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")

    # Symbiotic-specific nitrogen exchanges
    nitrogenase_fixation: float =                  declare(default=0., unit="mol.s-1", unit_comment="of amonium", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    
    mycorrhiza_infected_length: float =          declare(default=0., unit="m", unit_comment="of root segment infected by mycorrhiza", description="Length of the root segment infected by AMF", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")

    mycorrhizal_mediated_import_Nm: float =          declare(default=0., unit="mol.s-1", unit_comment="of amonium", description="Transfer of inorganic nitrogen from michoriza to root", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")

    # Deficits
    deficit_Nm: float = declare(default=0., unit="mol.s-1", unit_comment="of mineral nitrogen", description="Mineral nitrogen deficit rate in root", 
                                         min_value="", max_value="", value_comment="", references="Hypothesis of no initial deficit", DOI="",
                                          variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    deficit_AA: float = declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="Amino acids deficit rate in root", 
                                           min_value="", max_value="", value_comment="", references="Hypothesis of no initial deficit", DOI="",
                                            variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    deficit_AA_phloem: float = declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="Amino acids deficit rate in root phloem", 
                                           min_value="", max_value="", value_comment="", references="Hypothesis of no initial deficit", DOI="",
                                            variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    deficit_Nm_xylem: float = declare(default=0., unit="mol.s-1", unit_comment="of mineral nitrogen", description="Mineral nitrogen deficit rate in root", 
                                           min_value="", max_value="", value_comment="", references="Hypothesis of no initial deficit", DOI="",
                                            variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    deficit_AA_xylem: float = declare(default=0., unit="mol.s-1", unit_comment="of mineral nitrogen", description="Amino acids deficit rate in root", 
                                           min_value="", max_value="", value_comment="", references="Hypothesis of no initial deficit", DOI="",
                                            variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")

    # SUMMED STATE VARIABLES

    C_Nm_average: float =                   declare(default=3., unit="mol.g-1", unit_comment="of nitrates", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    C_AA_average: float =                   declare(default=1., unit="mol.g-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    C_hexose_average: float =               declare(default=0., unit="mol.g-1", unit_comment="of hexose", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    total_cytokinins: float =           declare(default=8.6, unit="UA", unit_comment="of cytokinins", description="",
                                                min_value="", max_value="", value_comment="", references="CN-Wheat", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    C_xylem_Nm_average: float =             declare(default=0., unit="mol", unit_comment="of nitrates", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    C_xylem_AA_average: float =             declare(default=0., unit="mol", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    C_phloem_AA_average: float =             declare(default=0., unit="mol", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    total_phloem_AA: float =            declare(default=-1, unit="mol", unit_comment="of amino acids", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    C_phloem_AA: float =            declare(default=10, unit="mol.m-3", unit_comment="of amino acids", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    Nm_root_to_shoot_xylem: float =        declare(default=0., unit="mol.h-1", unit_comment="of nitrates", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    AA_root_to_shoot_xylem: float =        declare(default=0., unit="mol.h-1", unit_comment="of amino acids", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    AA_root_to_shoot_phloem: float =       declare(default=0, unit="mol.time_step-1", unit_comment="of amino acids", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    total_AA_rhizodeposition: float =   declare(default=0., unit="mol.h-1", unit_comment="of amino acids", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    cytokinin_synthesis: float =        declare(default=0., unit=".s-1", unit_comment="of cytokinin", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    
    # Plotting utilities only
    simple_import_Nm: float =        declare(default=0., unit="mol.s-1", unit_comment="of nitrate", description="Total MM over the root system relative to cylinder surface to compare the current model with a simpler one", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    net_N_uptake: float =           declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                min_value=1e-11, max_value=1e-9, value_comment="", references="", DOI="",
                                                variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")
    net_mineral_N_uptake: float =           declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                min_value=1e-11, max_value=1e-9, value_comment="", references="", DOI="",
                                                variable_type="state_variable", by="model_nitrogen", state_variable_type="NonInertialExtensive", edit_by="user")

    
    # --- @note MODEL PARAMETERS INITIALIZATION ---

    # time resolution
    sub_time_step: int =                declare(default=3600, unit="s", unit_comment="", description="MUST be a multiple of base time_step", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")

    # N TRANSPORT PROCESSES
    # kinetic parameters
    
    vmax_HATS_Nm_amplitude: float =     declare(default=3.3863e-11, unit="mol.s-1.m-2", unit_comment="of nitrates", description="",
                                                min_value="", max_value="", value_comment="", references="Amplitude for Vmax parameter of Log-normal density fitting with Siddiqi et al. 1990", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    vmax_HATS_Nm_centering: float =     declare(default=-8.1648, unit="dimensionless", unit_comment="", description="",
                                                min_value="", max_value="", value_comment="", references="Mean for Vmax parameter of Log-normal density fitting with Siddiqi et al. 1990", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    vmax_HATS_Nm_spread: float =        declare(default=0.9917, unit="dimensionless", unit_comment="", description="",
                                                min_value="", max_value="", value_comment="", references="Variance for Vmax parameter of Log-normal density fitting with Siddiqi et al. 1990", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_HATS_Nm_amplitude: float =     declare(default=1.3137e-4, unit="mol.m-3", unit_comment="of nitrates", description="",
                                                min_value="", max_value="", value_comment="", references="Amplitude for Km parameter of Log-normal density fitting with Siddiqi et al. 1990", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_HATS_Nm_centering: float =     declare(default=-6.2277, unit="dimensionless", unit_comment="", description="",
                                                min_value="", max_value="", value_comment="", references="Mean for Km parameter of Log-normal density fitting with Siddiqi et al. 1990", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_HATS_Nm_spread: float =        declare(default=1.7539, unit="dimensionless", unit_comment="", description="",
                                                min_value="", max_value="", value_comment="", references="Variance for Km parameter of Log-normal density fitting with Siddiqi et al. 1990", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_LATS_Nm_decrease_slope: float =            declare(default= - 5.615e-7, unit="m.g.mol-1.s-1", unit_comment="m3.g.mol-1.m-2.s-1", description="Slope of linear decrease of LATS Km according to root symplasmic Nm concentration",
                                                min_value="", max_value="", value_comment="", references="Siddiqui 1990 showing best fit for linear model; Barillot et al. 2016, but diverging from publication as linear", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_LATS_Nm_origin: float =            declare(default=6.5026e-10, unit="m.s-1", unit_comment="m3.m-2.s-1 of nitrates", description="Origin value of linear decrease of LATS Km according to root symplasmic Nm concentration",
                                                min_value="", max_value="", value_comment="", references="Siddiqui 1990 showing best fit for linear model; Barillot et al. 2016, but diverging from publication as linear", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    vmax_Nm_xylem: float =              declare(default=1e-5, unit="mol.s-1.m-2", unit_comment="of nitrates", description="",
                                                min_value="", max_value="", value_comment="*10e2 from outside root as a lower surface has to compete with external surface and presents LATS", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_Nm_xylem: float =                declare(default=1e-3, unit="mol.g-1", unit_comment="of nitrates", description="",
                                                min_value="", max_value="", value_comment="adjusted to avoid accumulation in symplasm", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    vmax_AA_root: float =               declare(default=1e-8, unit="mol.s-1.m-2", unit_comment="of amino acids", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_AA_root: float =                 declare(default=1e-1, unit="mol.m-3", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    vmax_AA_xylem: float =              declare(default=1e-5, unit="mol.s-1.m-2", unit_comment="of amino acids", description="",
                                                min_value="", max_value="", value_comment="*10e2 from outside root as presents LATS only", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_AA_xylem: float =                declare(default=1e-1, unit="mol.g-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    diffusion_soil: float =             declare(default=2.5e-12, unit="g.s-1.m-2", unit_comment="of solute", description="", 
                                                min_value="", max_value="", value_comment="while there is no soil model balance", references="", DOI="", 
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    diffusion_xylem: float =            declare(default=1e-10, unit="g.s-1.m-2", unit_comment="of solute", description="",
                                                min_value="", max_value="", value_comment="from 1e-8, lowered to avoid crazy segment loading bugs", references="", DOI="", 
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    diffusion_phloem: float =           declare(default=1.2e-10 / 10000, unit="g.s-1.m-2", unit_comment="of solute", description="",
                                                min_value="", max_value="", value_comment="1.2e-8 * Important value to avoid harsh growth limitations", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="I", edit_by="user")
    vmax_unloading_AA_phloem: float = declare(default=1e-7 * 10, unit="mol.m-2.s-1", unit_comment="", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    km_unloading_AA_phloem: float = declare(default=100, unit="mol.m-3", unit_comment="", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    reference_rate_of_AA_consumption_by_growth: float = declare(default=1e-10, unit="mol.s-1.g-1", unit_comment="of hexose", description="Coefficient of permeability of unloading phloem", 
                                                min_value="", max_value="", value_comment="From RhizoDep parameter, applied 5e-13 * 6 * 12 / 0.44 * 0.015 / 14 / 1.4", references="Reference consumption rate of hexose for growth for a given root element (used to multiply the reference unloading rate when growth has consumed hexose)", DOI="",
                                                variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user")
    diffusion_apoplasm: float =         declare(default=1e-13, unit="g.s-1.m-2", unit_comment="of solute", description="", 
                                                min_value="", max_value="", value_comment="while there is no soil model balance", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    
    # Mycorrhiza related parameters
    # Infection-related
    mycorrhiza_infection_probability: float =   declare(default=0.15 * 100 / (24*3600) / 1e-3, unit=".m-1.s-1.g-1", unit_comment="", description="Probability of AMF primary infection by root length, time and fungus structural mass unit.", 
                                                min_value="", max_value="", value_comment="", references="(Schnepf et al. 2016)", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    mycorrhiza_max_distance_from_tip: float =   declare(default=0.15, unit="m", unit_comment="", description="Maximal distance from tip were AMF arbuscules have been observed", 
                                                min_value="", max_value="", value_comment="", references="(Schnepf et al. 2016)", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    mycorrhiza_internal_infection_speed: float =   declare(default=0.13 * 1e-2 / (24 * 3600), unit="m.s-1", unit_comment="", description="Secondary infection progression speed along root-length", 
                                                min_value="", max_value="", value_comment="", references="(Schnepf et al. 2016)", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    
    # Transport-related
    vmax_Nm_to_roots_fungus: float =   declare(default=2.55e-9 / 3600, unit="mol.s-1.m-1", unit_comment="", description="Maximal rate of amonium export from AMF to roots, per infected root length", 
                                                min_value="", max_value="", value_comment="Assuming a length per struct_mass ratio stabilised at 70 for plants at 1200 DD", references="Hawkins et al. 2000; 1999", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_Nm_to_roots_fungus: float =   declare(default=16.6e-6, unit="mol.g-1", unit_comment="per g of AMF", description="Affinity of ZmAMT3;1 for amonium during export from AMF to roots", 
                                                min_value="", max_value="", value_comment="Measured on yeast", references="Hui et al. 2022", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")

    # Biological Nitrogen Fixation related parameters
    vmax_bnf: float =   declare(default=31.e-6 * 2 / 3600, unit="mol.g-1.s-1", unit_comment="g-1 of Nodule dry mass", description="Maximal nitrogenase Biological Nitrogen Fixation (BNF) rate observed in the litterature from ARA", 
                                min_value="", max_value="", value_comment="", references="Lyu et al. 2020; Soper et al. 2021", DOI="",
                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    
    K_bnf_Nm_inibition: float =   declare(default=1e-4, unit="mol.g-1", unit_comment="", description="Inibition affinity for mineral Nitrogen in Michaelis-Menten formalism", 
                                min_value="", max_value="", value_comment="", references="Assumption of inibition starting at low Nm concentration", DOI="",
                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    
    Km_hexose_bnf: float =   declare(default=1e-5, unit="mol.g-1", unit_comment="", description="Hexose affinity by consumption by N2 reduction chain of reactions",
                                min_value="", max_value="", value_comment="", references="Udvardi et Day 1997", DOI="",
                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    
    # metabolism-related parameters
    transport_C_regulation: float =     declare(default=7e-3/7, unit="mol.g-1", unit_comment="of hexose", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")

    # N METABOLISM PROCESSES
    # kinetic parameters
    smax_AA: float =                    declare(default=1e-5, unit="mol.s-1.g-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="*100 from ref to come closer to the 30% prop in whole synthesis expected", references="(Barillot 2016)", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_Nm_AA: float =                   declare(default=350e-6 * 100, unit="mol.g-1", unit_comment="of nitrates", description="", 
                                                min_value="", max_value="", value_comment="Changed to increase differences uppon Nm changes", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_C_AA: float =                    declare(default=350e-6, unit="mol.g-1", unit_comment="of hexose", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    smax_struct: float =                declare(default=0., unit="mol.s-1.g-1", unit_comment="of structure", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_AA_struct: float =               declare(default=250e-6, unit="mol.g-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    smax_stor: float =                  declare(default=0., unit="mol.s-1.g-1", unit_comment="of storage", description="",
                                                min_value="", max_value="", value_comment="from 1e-10, 0 for wheat", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_AA_stor: float =                 declare(default=250e-6, unit="mol.g-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    cmax_stor: float =                  declare(default=1e-9, unit="mol.s-1.g-1", unit_comment="of storage", description="", 
                                                min_value="", max_value="", value_comment="Supposing no priority between storage and unstorage, unless when catabolism is downregulated ", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_stor_catab: float =              declare(default=250e-6, unit="mol.g-1", unit_comment="of storage", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    cmax_AA: float =                    declare(default=5e-9 / 10, unit="mol.s-1.g-1", unit_comment="of amino acids", description="",
                                                min_value="", max_value="", value_comment="5e-9 for now not relevant as it doesn't contribute to C_hexose_root balance.", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_AA_catab: float =                declare(default=2.5e-6 * 1e3, unit="mol.g-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    storage_C_regulation: float =       declare(default=3e1, unit="mol.g-1", unit_comment="of hexose", description="", 
                                                min_value="", max_value="", value_comment="Changed to avoid reaching Vmax with slight decrease in hexose content", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    
    # HORMONES METABOLISM PROCESSES
    # kinetic parameters
    smax_cytok: float =                 declare(default=9e-4, unit="UA.s-1.g-1", unit_comment="of cytokinins", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_C_cytok: float =                 declare(default=1.2e-3, unit="UA.g-1", unit_comment="of hexose", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_N_cytok: float =                 declare(default=5.0e-5, unit="mol.g-1", unit_comment="of nitrates", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")

    # Temperature-related parameters
    # Active processes, Q10 bell-shaped dependancy
    active_processes_T_ref: float = declare(default=20, unit="°C", unit_comment="", description="the reference temperature", 
                                               min_value="", max_value="", value_comment="", references="Most measured kinetics have been performed in laboratory conditions and hydroponics", DOI="",
                                                variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user")
    active_processes_A: float = declare(default=-0.0442, unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", 
                                           min_value="", max_value="", value_comment="", references="Gifford (1995), see T_ref", DOI="",
                                                variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user")
    active_processes_B: float = declare(default=1.55, unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", 
                                           min_value="", max_value="", value_comment="", references="Gifford (1995), see T_ref", DOI="",
                                                variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user")
    active_processes_C: float = declare(default=1, unit="adim", unit_comment="", description="parameter C (either 0 or 1)", 
                                           min_value="", max_value="", value_comment="", references="Gifford (1995), see T_ref", DOI="",
                                                variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user")
    
    # Passive processes, supposing no temperature dependancy
    passive_processes_T_ref: float = declare(default=20, unit="°C", unit_comment="", description="the reference temperature", 
                                             min_value="", max_value="", value_comment="", references="We assume that the permeability does not directly depend on temperature, according to the contrasted results obtained by Wan et al. (2001) on poplar, Shen and Yan (2002) on crotalaria, Hill et al. (2007) on wheat, or Kaldy (2012) on a sea grass.", DOI="",
                                                variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user")
    passive_processes_A: float = declare(default=0., unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", 
                                         min_value="", max_value="", value_comment="", references="see T_ref", DOI="",
                                                variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user")
    passive_processes_B: float = declare(default=1., unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", 
                                         min_value="", max_value="", value_comment="", references="see T_ref", DOI="",
                                                variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user")
    passive_processes_C: float = declare(default=0., unit="adim", unit_comment="", description="parameter C (either 0 or 1)", 
                                         min_value="", max_value="", value_comment="", references="see T_ref", DOI="",
                                                variable_type="parameter", by="model_carbon", state_variable_type="", edit_by="user")

    # CONVERSION RATIO FOR STATE VARIABLES
    r_Nm_AA: float =                    declare(default=1.4, unit="adim", unit_comment="concentration ratio", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    r_AA_struct: float =                declare(default=65, unit="adim", unit_comment="concentration ratio", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    r_AA_stor: float =                  declare(default=65, unit="adim", unit_comment="concentration ratio", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    struct_mass_N_content: float = declare(default=0.015 / 14, unit="mol.g-1", unit_comment="of nitrogen", description="C content of structural mass", 
                                                    min_value="", max_value="", value_comment="", references="We assume that the structural mass contains 1.5% of N. (Barillot et al. 2016)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    
    # Helpers to keep labels intergers
    label_Segment: int = declare(default=1, unit="adim", unit_comment="", description="label utility", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    label_Apex: int = declare(default=2, unit="adim", unit_comment="", description="label utility", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    
    
    # Helpers to keep types intergers
    type_Base_of_the_root_system: int = declare(default=1, unit="adim", unit_comment="", description="type utility", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    type_Support_for_seminal_root: int = declare(default=2, unit="adim", unit_comment="", description="type utility", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    type_Seminal_root_before_emergence: int = declare(default=3, unit="adim", unit_comment="", description="type utility", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    type_Support_for_adventitious_root: int = declare(default=4, unit="adim", unit_comment="", description="type utility", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    type_Adventitious_root_before_emergence: int = declare(default=5, unit="adim", unit_comment="", description="type utility", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    type_Normal_root_before_emergence: int = declare(default=6, unit="adim", unit_comment="", description="type utility", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    type_Normal_root_after_emergence: int = declare(default=7, unit="adim", unit_comment="", description="type utility", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    type_Stopped: int = declare(default=8, unit="adim", unit_comment="", description="type utility", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    type_Just_stopped: int = declare(default=9, unit="adim", unit_comment="", description="type utility", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    type_Dead: int = declare(default=10, unit="adim", unit_comment="", description="type utility", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    type_Just_dead: int = declare(default=11, unit="adim", unit_comment="", description="type utility", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    type_Root_nodule: int = declare(default=12, unit="adim", unit_comment="", description="type utility", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")

    # DEFINE WHICH SOLUTES WILL BE TRANSPORTED IN CONDUCTIVE ELEMENTS
    solute_configs = {
    "xylem_Nm": {
        "solute_massic_concentration_prop": "xylem_Nm",
        "conductive_element_volume_prop": "xylem_volume",
        "water_flux_prop": "axial_export_water_up_xylem",
        "radial_solute_flux": lambda n: n.export_Nm - n.apoplastic_Nm_soil_xylem - n.diffusion_Nm_xylem,
        "flux_shoot_boundary": lambda props: props["Nm_input_rate_xylem"][1],
        "boundary_shoot_solute_concentration": lambda props: props["Cv_Nm_xylem_collar"][1],
        "solute_flux_to_shoot": "Nm_root_to_shoot_xylem",
        "solute_volumic_concentration_bounds": (1e-3, 5e2),
        },
    "xylem_AA": {
        "solute_massic_concentration_prop": "xylem_AA",
        "conductive_element_volume_prop": "xylem_volume",
        "water_flux_prop": "axial_export_water_up_xylem",
        "radial_solute_flux": lambda n: n.export_AA - n.apoplastic_AA_soil_xylem,
        "flux_shoot_boundary": lambda props: props["AA_input_rate_xylem"][1],
        "boundary_shoot_solute_concentration": lambda props: props["Cv_AA_xylem_collar"][1],
        "solute_flux_to_shoot": "AA_root_to_shoot_xylem",
        "solute_volumic_concentration_bounds": (1e-4, 5e2),
        },
    "phloem_AA": {
        "solute_massic_concentration_prop": "phloem_AA",
        "conductive_element_volume_prop": "phloem_volume",
        "water_flux_prop": "axial_export_water_up_phloem",
        "radial_solute_flux": lambda n: -n.diffusion_AA_phloem - n.unloading_AA_phloem,
        "flux_shoot_boundary": lambda props: props["AA_input_rate_phloem"][1],
        "boundary_shoot_solute_concentration": lambda props: props["Cv_AA_phloem_collar"][1],
        "solute_flux_to_shoot": "AA_root_to_shoot_phloem",
        "solute_volumic_concentration_bounds": (10, 2e3),
        }
    }


    def __init__(self, g, time_step, **scenario) -> None:

        """
        Description
        Initialization of nitrogen-related variables

        Parameters
        :param g: MTG
        """
        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)
        

        self.g = g
        self.props = self.g.properties()
        self.time_step = time_step
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step, data=self.props)
        self.vertices = self.g.vertices(scale=self.g.max_scale())

        self.link_self_to_mtg()


    # @note PROCESSES OF N TRANSPORT AND METABOLISM

    # RADIAL TRANSPORT PROCESSES
    # MINERAL NITROGEN TRANSPORT
    @rate
    def _import_Nm(self, Nm, soil_Nm, root_exchange_surface, soil_temperature, C_hexose_root=1e-4):
        """
                Description
                ___________
                Nitrogen transport between local soil, local root segment and global vessels (xylem and phloem).

                Parameters
                __________
                :param Km_Nm_root: Active transport from soil Km parameter (mol.m-3)
                :param vmax_Nm_emergence: Surfacic maximal active transport rate in roots (mol.m-2.s-1)
                :param Km_Nm_xylem: Active transport from root Km parameter (mol.m-3)
                :param diffusion_phloem: Mineral nitrogen diffusion parameter (m.s-1)
                :param transport_C_regulation: Km coefficient for the nitrogen active transport regulation function
                by root C (mol.g-1) (?)
                by root mineral nitrogen (mol.m-3)

                Hypothesis
                __________
                H1: We summarize radial active transport controls (transporter density, affinity regulated with genetics
                or environmental control, etc) as one mean transporter following Michaelis Menten's model.

                H2: We can summarize apoplastic and symplastic radial transport through one radial transport.
                Differentiation with epidermis conductance loss, root hair density, aerenchyma, etc, is supposed to impact Vmax.

                H3: We declare similar kinetic parameters for soil-root and root-xylem active transport (exept for concentration conflict)
                """
        
        # Log normal dependancy is used to account for observation of inducted HATS (iHATS) in addition to consititutive HATS (cHATS) already present
        # With a shift for HATS in the low concentration domain from high affinity-low vmax to low affinity-high vmax
        vmax_HATS_Nm_root = self.root_nitrate_lognorm_regulation(Nm, self.vmax_HATS_Nm_amplitude,
                                                                      self.vmax_HATS_Nm_centering,
                                                                      self.vmax_HATS_Nm_spread)
        
        Km_HATS_Nm_root = self.root_nitrate_lognorm_regulation(Nm, self.Km_HATS_Nm_amplitude,
                                                                      self.Km_HATS_Nm_centering,
                                                                      self.Km_HATS_Nm_spread)
        
        import_Nm_HATS = soil_Nm * vmax_HATS_Nm_root / (soil_Nm + Km_HATS_Nm_root)

        # Then we account for low affinity transporters which account for a large part of the uptake in high concentration domains
        # Km_LATS_Nm_root = self.Km_Nm_root_LATS * self.Km_LATS_Nm_slope_modifier * np.exp( - self.Km_LATS_Nm_regulation_speed * Nm) # TODO : Ask Romain if also chosen out of Siddiqi et al. 1990
        Km_LATS_Nm_root = np.maximum(0., self.Km_LATS_Nm_decrease_slope * Nm + self.Km_LATS_Nm_origin) #: Rate constant for nitrates influx at High soil N concentration; LATS linear phase
        
        import_Nm_LATS = Km_LATS_Nm_root * soil_Nm

        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        temperature_modification = self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)
        
        carbon_regulation = (C_hexose_root / (C_hexose_root + self.transport_C_regulation))

        return (import_Nm_HATS + import_Nm_LATS) * temperature_modification * root_exchange_surface * carbon_regulation
    

    @rate
    def _import_Nm_LATS(self, Nm, soil_Nm, root_exchange_surface, soil_temperature, C_hexose_root=1e-4):
        """
                Description
                ___________
                Nitrogen transport between local soil, local root segment and global vessels (xylem and phloem).

                Parameters
                __________
                :param Km_Nm_root: Active transport from soil Km parameter (mol.m-3)
                :param vmax_Nm_emergence: Surfacic maximal active transport rate in roots (mol.m-2.s-1)
                :param Km_Nm_xylem: Active transport from root Km parameter (mol.m-3)
                :param diffusion_phloem: Mineral nitrogen diffusion parameter (m.s-1)
                :param transport_C_regulation: Km coefficient for the nitrogen active transport regulation function
                by root C (mol.g-1) (?)
                by root mineral nitrogen (mol.m-3)

                Hypothesis
                __________
                H1: We summarize radial active transport controls (transporter density, affinity regulated with genetics
                or environmental control, etc) as one mean transporter following Michaelis Menten's model.

                H2: We can summarize apoplastic and symplastic radial transport through one radial transport.
                Differentiation with epidermis conductance loss, root hair density, aerenchyma, etc, is supposed to impact Vmax.

                H3: We declare similar kinetic parameters for soil-root and root-xylem active transport (exept for concentration conflict)
                """
        
        # Then we account for low affinity transporters which account for a large part of the uptake in high concentration domains
        # Km_LATS_Nm_root = self.Km_Nm_root_LATS * self.Km_LATS_Nm_slope_modifier * np.exp( - self.Km_LATS_Nm_regulation_speed * Nm) # TODO : Ask Romain if also chosen out of Siddiqi et al. 1990
        Km_LATS_Nm_root = np.maximum(0., self.Km_LATS_Nm_decrease_slope * Nm + self.Km_LATS_Nm_origin) #: Rate constant for nitrates influx at High soil N concentration; LATS linear phase
        
        import_Nm_LATS = Km_LATS_Nm_root * soil_Nm

        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        temperature_modification = self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)
        
        carbon_regulation = (C_hexose_root / (C_hexose_root + self.transport_C_regulation))

        return import_Nm_LATS * temperature_modification * root_exchange_surface * carbon_regulation


    def root_nitrate_lognorm_regulation(self, x, A, mu, sigma):
        result = A / (x * sigma * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
        return np.where((np.isnan(result)) | (np.isinf(result)), 0., np.maximum(result, 0.))
        

    @rate
    def _diffusion_Nm_soil(self, Nm, soil_Nm, root_exchange_surface, living_struct_mass, symplasmic_volume, soil_temperature):
        
        # Passive radial diffusion between soil and cortex.
        # It happens only through root segment external surface.
        # We summarize apoplasm-soil and cortex-soil diffusion in 1 flow.
        diffusion_soil = self.diffusion_soil * self.temperature_modification(soil_temperature=soil_temperature,
                                                                    T_ref=self.passive_processes_T_ref,
                                                                    A=self.passive_processes_A,
                                                                    B=self.passive_processes_B,
                                                                    C=self.passive_processes_C)
        
        return np.where(symplasmic_volume <= 0., 0.,
                        (diffusion_soil * ((Nm * living_struct_mass / np.where(symplasmic_volume <= 0., 1., symplasmic_volume)) - soil_Nm) * root_exchange_surface)) # Nested where to safegard from / 0 , does not yield in results

    @rate
    def _export_Nm(self, Nm, xylem_exchange_surface, soil_temperature, C_hexose_root=1e-4):
        # We define active export to xylem from root segment
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        vmax_Nm_xylem = self.vmax_Nm_xylem * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)
        return ((Nm * vmax_Nm_xylem) / (Nm + self.Km_Nm_xylem)) * xylem_exchange_surface * (
                C_hexose_root / (C_hexose_root + self.transport_C_regulation))

    @rate
    def _diffusion_Nm_xylem(self, xylem_Nm, Nm, xylem_exchange_surface, soil_temperature, living_struct_mass, symplasmic_volume, xylem_volume):
        # Passive radial diffusion between xylem and cortex through plasmalema
        diffusion_xylem = self.diffusion_xylem * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.passive_processes_T_ref,
                                                                     A=self.passive_processes_A,
                                                                     B=self.passive_processes_B,
                                                                     C=self.passive_processes_C)

        # if debug: print((xylem_Nm * living_struct_mass / xylem_volume), (Nm * living_struct_mass / symplasmic_volume))
        return diffusion_xylem * ((xylem_Nm * living_struct_mass / xylem_volume) - (Nm * living_struct_mass / symplasmic_volume)) * xylem_exchange_surface

    @rate
    def _apoplastic_Nm_soil_xylem(self, import_Nm, diffusion_Nm_soil, soil_Nm, xylem_Nm, radius, radial_import_water_xylem_apoplastic, length, xylem_differentiation_factor, endodermis_conductance_factor, living_struct_mass, xylem_volume, soil_temperature):
                
        # advection_process = - (xylem_Nm * living_struct_mass / xylem_volume) * radial_import_water_xylem_apoplastic # accounts for xylem opening and endodermis conductance already
        advection_process = np.where(radial_import_water_xylem_apoplastic > 0, - soil_Nm * radial_import_water_xylem_apoplastic, # Here we compure a flux leaving the segment, but here it enters
                                     0.)# Since we don't account for apoplasm, in this situation instead of a direct outflow to soil, we expect that this would be reuptaken by the root
        
        # Direct diffusion between soil and xylem when 1) xylem is apoplastic and 2) endoderm is not differentiated
        # Here, surface is not really representative of a structure as everything is apoplasmic
        diffusion_apoplasm = self.diffusion_apoplasm * self.temperature_modification(soil_temperature=soil_temperature,
                                                                T_ref=self.passive_processes_T_ref,
                                                                A=self.passive_processes_A,
                                                                B=self.passive_processes_B,
                                                                C=self.passive_processes_C)
        diffusion_process = diffusion_apoplasm * (xylem_Nm * living_struct_mass / np.where(xylem_volume <=0, 1., xylem_volume) - soil_Nm) * 2 * np.pi * radius * length * xylem_differentiation_factor * endodermis_conductance_factor

        return np.where((xylem_volume <= 0.) | (endodermis_conductance_factor == 0), 0.,
                        advection_process + diffusion_process)


    # AMINO ACID TRANSPORT
    @rate
    def _import_AA(self, soil_AA, root_exchange_surface, soil_temperature, C_hexose_root=1e-4):
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        vmax_AA_root = self.vmax_AA_root * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)
        return ((soil_AA * vmax_AA_root / (soil_AA + self.Km_AA_root)) * root_exchange_surface * (
            C_hexose_root / (C_hexose_root + self.transport_C_regulation)))

    @rate
    def _diffusion_AA_soil(self, AA, soil_AA, root_exchange_surface, living_struct_mass, symplasmic_volume, soil_temperature):

        # We define amino acid passive diffusion to soil
        diffusion_soil = self.diffusion_soil * self.temperature_modification(soil_temperature=soil_temperature,
                                                                    T_ref=self.passive_processes_T_ref,
                                                                    A=self.passive_processes_A,
                                                                    B=self.passive_processes_B,
                                                                    C=self.passive_processes_C)
        
        return np.where(symplasmic_volume <= 0., 0.,
                        (diffusion_soil * ((AA * living_struct_mass / np.where(symplasmic_volume <= 0., 1., symplasmic_volume)) - soil_AA) * root_exchange_surface ))        


    @rate
    def _export_AA(self, AA, xylem_exchange_surface, soil_temperature, C_hexose_root=1e-4):
        # We define active export to xylem from root segment
        # Km is defined as a constant here
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        vmax_AA_xylem = self.vmax_AA_xylem * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)
        return ((AA * vmax_AA_xylem / (AA + self.Km_AA_xylem))
                * xylem_exchange_surface * (C_hexose_root / (C_hexose_root + self.transport_C_regulation)))

    @rate
    def _apoplastic_AA_soil_xylem(self, import_AA, diffusion_AA_soil, soil_AA, xylem_AA, radius, length, radial_import_water_xylem_apoplastic, xylem_differentiation_factor, endodermis_conductance_factor, living_struct_mass, xylem_volume, soil_temperature):

        # If water is imported from the soil
        advection_process = np.where(radial_import_water_xylem_apoplastic > 0, - soil_AA * radial_import_water_xylem_apoplastic, # Here we compure a flux leaving the segment, but here it enters
                                     0.) # Since we don't account for apoplasm, in this situation instead of a direct outflow to soil, we expect that this would be reuptaken by the root
        # advection_process = - (xylem_AA * living_struct_mass / xylem_volume) * radial_import_water_xylem_apoplastic # accounts for xylem opening and endodermis conductance already

        # Direct diffusion between soil and xylem when 1) xylem is apoplastic and 2) endoderm is not differentiated
        diffusion_apoplasm = self.diffusion_apoplasm * self.temperature_modification(soil_temperature=soil_temperature,
                                                                T_ref=self.passive_processes_T_ref,
                                                                A=self.passive_processes_A,
                                                                B=self.passive_processes_B,
                                                                C=self.passive_processes_C)
        diffusion_process = diffusion_apoplasm * (xylem_AA * living_struct_mass / np.where(xylem_volume <= 0., 1., xylem_volume) - soil_AA) * 2 * np.pi * radius * length * xylem_differentiation_factor * endodermis_conductance_factor

        return np.where((xylem_volume <= 0) | (endodermis_conductance_factor == 0), 0.,
                        advection_process + diffusion_process)
            
            
    @rate
    def _diffusion_AA_phloem(self, hexose_consumption_by_growth, AA, phloem_AA, phloem_exchange_surface, soil_temperature, living_struct_mass, symplasmic_volume, phloem_volume):
        """ Passive radial diffusion between phloem and cortex through plasmodesmata """
        Cv_AA_phloem = (phloem_AA * living_struct_mass) / np.where(phloem_volume <=0., 1., phloem_volume)

        AA_consumption_by_growth = (hexose_consumption_by_growth * 6 * 12 / 0.44) * self.struct_mass_N_content / self.r_Nm_AA

        diffusion_phloem = self.diffusion_phloem * (1 + (AA_consumption_by_growth / living_struct_mass) / self.reference_rate_of_AA_consumption_by_growth)

        diffusion_phloem *= self.temperature_modification(soil_temperature=soil_temperature,
                                                                    T_ref=self.passive_processes_T_ref,
                                                                    A=self.passive_processes_A,
                                                                    B=self.passive_processes_B,
                                                                    C=self.passive_processes_C)

        return np.where((phloem_volume <= 0.) | (symplasmic_volume <= 0.) | (Cv_AA_phloem <= (AA * living_struct_mass) / symplasmic_volume / 2.), 0.,
                        diffusion_phloem * (np.maximum(0, (phloem_AA * living_struct_mass) / np.where(phloem_volume <= 0., 1., phloem_volume)) 
                                            - np.maximum(0, (AA * living_struct_mass) / np.where(symplasmic_volume <= 0., 1., symplasmic_volume))) * phloem_exchange_surface)


    @rate
    def _unloading_AA_phloem(self, phloem_AA, hexose_consumption_by_growth, phloem_exchange_surface, soil_temperature, living_struct_mass, phloem_volume, symplasmic_volume):
        Cv_AA_phloem = (phloem_AA * living_struct_mass) / phloem_volume
        
        # if Cv_AA_phloem <= (AA * living_struct_mass) / symplasmic_volume / 2.:
        #         return 0
        # else:
        AA_consumption_by_growth = (hexose_consumption_by_growth * 6 * 12 / 0.44) * self.struct_mass_N_content / self.r_Nm_AA
        vmax_unloading_AA_phloem = self.vmax_unloading_AA_phloem * (1 + (AA_consumption_by_growth / living_struct_mass) / self.reference_rate_of_AA_consumption_by_growth)
        vmax_unloading_AA_phloem *= self.temperature_modification(soil_temperature=soil_temperature,
                                                            T_ref=self.active_processes_T_ref,
                                                            A=self.active_processes_A,
                                                            B=self.active_processes_B,
                                                            C=self.active_processes_C)
        
        return np.minimum(vmax_unloading_AA_phloem * Cv_AA_phloem * phloem_exchange_surface / (
                    self.km_unloading_AA_phloem + phloem_AA), phloem_AA * living_struct_mass / 2)


    @axial
    @rate
    def _axial_transport_N(self):
        """
        Transient resolution of solute advection
        Rewoked to rely on external definition of solutes and related properties, so we ensure it is easily appended
        """
        # t1 = time.time()
        g = self.g
        props = g.properties()
        struct_mass = g.property('struct_mass')

        # Set up local vids
        local_vid = 0
        local_vids = {}
        for vid, value in struct_mass.items():
            if value > 0:
                local_vids[vid] = local_vid
                local_vid += 1

        living_struct_mass = np.array([g.node(v).living_struct_mass for v in local_vids])

        # Create a solute configs and buffer that enables iterating through vertices only once
        solute_configs = self.solute_configs
        solute_buffers = {
            name: {"solute_amount": [], "conductive_element_volume": [], "radial_solute_flux": [], "boundary_solute_flux_from_shoot": 0, "impacted_by_root_shoot_boundary": {},
                "row": [], "col": [], "data": []}
            for name in solute_configs
        }

        elt_number = len(local_vids)
        # print("start building the axial transport matrix...")
        for v in self.vertices:
            n = g.node(v)
            if n.struct_mass > 0:
                lid = local_vids[v]

                for name, cfg in solute_configs.items():

                    buf = solute_buffers[name]
                    water_flux_prop = cfg["water_flux_prop"]
                    conductive_element_volume_prop = cfg["conductive_element_volume_prop"]
                    solute_massic_concentration_prop = cfg["solute_massic_concentration_prop"]
                    axial_diffusivity = 5e-8 # m2/s

                    water_flux = getattr(n, water_flux_prop)
                    conductive_element_volume = getattr(n, conductive_element_volume_prop)
                    solute_massic_concentration = getattr(n, solute_massic_concentration_prop)
                    solute_amount = solute_massic_concentration * n.living_struct_mass
                    solute_volumic_concentration =  solute_amount / conductive_element_volume
                    buf["solute_amount"].append(solute_amount)
                    buf["conductive_element_volume"].append(conductive_element_volume)
                    buf["radial_solute_flux"].append(cfg["radial_solute_flux"](n))

                    # Diffusive component
                    if v in self.collar_children:
                        parent = 1
                    else:
                        parent = g.parent(v)

                    if parent is not None:
                        dx = (n.length + g.node(parent).length) / 2
                        cross_area = np.pi * (0.1 * (n.radius + g.node(parent).radius) / 2)**2
                        D = axial_diffusivity * cross_area / dx
                        buf["row"].extend([lid, lid, local_vids[parent], local_vids[parent]])
                        buf["col"].extend([lid, local_vids[parent], local_vids[parent], lid])
                        buf["data"].extend([-D, D, -D, D])

                    # Advection component
                    if v in self.collar_children:
                        parent=1
                    else:
                        parent = g.parent(v)

                    if parent is None:
                        # This is imposed by data so there is no condition to apply this
                        if cfg["flux_shoot_boundary"](props) is not None:
                            buf["boundary_solute_flux_from_shoot"] = cfg["flux_shoot_boundary"](props)
                        else:
                            if water_flux < 0:
                                buf["boundary_solute_flux_from_shoot"] = - water_flux * cfg["boundary_shoot_solute_concentration"](props)
                            else:
                                buf["boundary_solute_flux_from_shoot"] = - water_flux * solute_volumic_concentration
                            # else condition to write only if bellow for system mass adjustment is replaced
                    else:
                        if water_flux > 0:
                            buf["row"].extend([lid, local_vids[parent]])
                            buf["col"].extend([lid, lid])
                            buf["data"].extend([-water_flux, water_flux])
                        else:
                            buf["row"].extend([local_vids[parent], lid])
                            buf["col"].extend([local_vids[parent], local_vids[parent]])
                            buf["data"].extend([water_flux, -water_flux])

                    # Identify which segments are directly impacted by collar flux
                    if v == 1:
                        # print("implicit concentration : ", buf["boundary_solute_flux_from_shoot"] / np.abs(water_flux))
                        impacted_volume = np.abs(water_flux) * self.time_step
                        cummulated_volume = conductive_element_volume
                        impacted_by_root_shoot_boundary = [local_vids[v]]
                        impacted_by_root_shoot_boundary_volume = [conductive_element_volume]
                        parent_set = [v]
                        while impacted_volume > cummulated_volume:
                            children_set = []
                            for parent in parent_set:
                                if parent == 1:
                                    children_set += self.collar_children
                                else:
                                    children_set += g.children(parent)
                            
                            if len(children_set) == 0:
                                # For very young root systems it is possble it exceeds whole root system's volume, so we prevent infinit loop
                                break

                            for cid in children_set:
                                children_conductive_element_volume = getattr(g.node(cid), conductive_element_volume_prop)
                                if children_conductive_element_volume is not None:
                                    if children_conductive_element_volume > 0:
                                        impacted_by_root_shoot_boundary.append(local_vids[cid])
                                        impacted_by_root_shoot_boundary_volume.append(children_conductive_element_volume)
                                        cummulated_volume += children_conductive_element_volume
                                
                            parent_set = children_set

                        total_grouped_volume = sum(impacted_by_root_shoot_boundary_volume)
                        impacted_by_root_shoot_boundary_prop = {impacted_by_root_shoot_boundary[k]: v/total_grouped_volume for k, v in enumerate(impacted_by_root_shoot_boundary_volume)}
                        buf["impacted_by_root_shoot_boundary"] = impacted_by_root_shoot_boundary_prop

        # Identity matrix (I)
        I = identity(elt_number, format="csc")

        # Solve sequentially for each solute
        for name, cfg in solute_configs.items():
                
            buf = solute_buffers[name]

            # Static components
            A = csc_matrix((buf["data"], (buf["row"], buf["col"])), shape=(elt_number, elt_number))
            V = np.array(buf["conductive_element_volume"])
            R = np.array(buf["radial_solute_flux"])
            B = buf["boundary_solute_flux_from_shoot"]
            impacted_by_root_shoot_boundary = buf["impacted_by_root_shoot_boundary"]

            # Initial conditions
            ns0 = np.array(buf["solute_amount"])

            boundary = np.zeros_like(R)
            for i, p in impacted_by_root_shoot_boundary_prop.items():
                boundary[i] = p * B

            # Recording what is applied (mol.s-1)
            props[cfg["solute_flux_to_shoot"]][1] = - boundary.sum()

            R_total = R + boundary

            # print("A sum", A.sum()) # YOU HAVE TO ENSURE IT NEARS 0 (<1e-30)
            A_to_V = A @ diags(1.0 / V)

            # SPLU
            LHS      = I - self.time_step * A_to_V          # shape (n×n), still sparse
            solve_BE = linalg.factorized(LHS)
            RHS   = ns0 + self.time_step * R_total
            n_sol = solve_BE(RHS)

            c_min, c_max = cfg["solute_volumic_concentration_bounds"]
            n_min = c_min * V
            n_max = c_max * V

            n_sol_clip = np.clip(n_sol, n_min, n_max)
            n_sol_clip = (ns0 + n_sol_clip) / 2 # Average the two resulting concentrations for numerical stability and repartition based on deficit
            deficit = (ns0 + R_total - n_sol_clip).sum()
            if deficit > 0:
                free = c_max * V - n_sol_clip
                if free.sum() < deficit:
                    print(name, "Warning impossible adjustment of concentrations")
                    # n_sol_clip = n_sol
                else:
                    n_sol_clip += deficit * free / free.sum()
            else:
                free = n_sol_clip - c_min * V
                if free.sum() < -deficit:
                    print(name, "Warning impossible adjustment of concentrations")
                    # n_sol_clip = n_sol
                else:
                    n_sol_clip += deficit * free / free.sum()

            # Debug print
            # print(name, "mass", n_sol_clip.sum() - (ns0.sum() + self.time_step * R_total.sum()))
            # print(name, n_sol_clip / V)

            # Retreive resulting massic concentrations and update MTG props with it
            Cm_sol = n_sol_clip / living_struct_mass
            # Quality check
            assert not np.any(Cm_sol < 0)
            props[cfg["solute_massic_concentration_prop"]].update(dict(zip(local_vids.keys(), Cm_sol)))

        # t2 = time.time()
        # print("axial time", t2 - t1)


    # METABOLIC PROCESSES
    @rate
    def _AA_synthesis(self, living_struct_mass, Nm, soil_temperature, C_hexose_root=1e-4):
        # amino acid synthesis
        smax_AA = self.smax_AA * self.temperature_modification(soil_temperature=soil_temperature,
                                                                    T_ref=self.active_processes_T_ref,
                                                                    A=self.active_processes_A,
                                                                    B=self.active_processes_B,
                                                                    C=self.active_processes_C)

        return np.where((C_hexose_root <= 0) | (Nm <= 0), 0.,
                        living_struct_mass * smax_AA / (
                    ((1 + self.Km_Nm_AA) / np.where(Nm <= 0., 1., Nm)) + ((1 + self.Km_C_AA) / np.where(C_hexose_root <= 0., 1., C_hexose_root))))
        

    @rate
    def _storage_synthesis(self, living_struct_mass, AA, soil_temperature):
        # Organic storage synthesis (Michaelis-Menten kinetic)
        smax_stor = self.smax_stor * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)
        return living_struct_mass * (smax_stor * AA / (self.Km_AA_stor + AA))


    @rate
    def _storage_catabolism(self, living_struct_mass, storage_protein, soil_temperature, C_hexose_root=1e-4):
        # Organic storage catabolism through proteinase
        Km_stor_root = self.Km_stor_catab * np.exp(self.storage_C_regulation * C_hexose_root)
        cmax_stor = self.cmax_stor * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)
        return living_struct_mass * cmax_stor * storage_protein / (Km_stor_root + storage_protein)


    @rate
    def _AA_catabolism(self, living_struct_mass, AA, soil_temperature, C_hexose_root=1e-4):
        # AA catabolism through GDH
        Km_stor_root = self.Km_AA_catab * np.exp(self.storage_C_regulation * C_hexose_root)
        cmax_AA = self.cmax_AA * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)
        return living_struct_mass * cmax_AA * AA / (Km_stor_root + AA)


    @rate
    def _nitrogenase_fixation(self, type, living_struct_mass, C_hexose_root, Nm, soil_temperature):

        # We model nitrogenase expression repression by higher nitrogen availability through an inibition law
        vmax_bnf = (self.vmax_bnf / (1 + (Nm / self.K_bnf_Nm_inibition))) * self.temperature_modification(soil_temperature=soil_temperature,
                                                                                                        T_ref=self.active_processes_T_ref,
                                                                                                        A=self.active_processes_A,
                                                                                                        B=self.active_processes_B,
                                                                                                        C=self.active_processes_C)
        # Michaelis-Menten formalism
        return np.where(type != self.type_Root_nodule, 0., 
                        living_struct_mass * vmax_bnf * C_hexose_root / (self.Km_hexose_bnf + C_hexose_root))
        

    # @state
    def _mycorrhiza_infected_length(self, vertex_index, mycorrhiza_infected_length, distance_from_tip, struct_mass_fungus, length):
        """
        From Scnepf et al 2016, modified with distance from tip here to avoid not computed root age
        NOTE : When coupling with a carbon model, mycorrhiza_infection_probability could be plasticized by mycorrhiza and C signaling exudation
        NOTE : When coupling with a carbon model, mycorrhiza_internal_infection_speed could be plasticized by C allocation to mycorrhiza.
        """

        if mycorrhiza_infected_length < length:
            # If a progress of infection is still possible
            local_infection_probability = np.maximum(1 - distance_from_tip / self.mycorrhiza_max_distance_from_tip, 0.) * (
                self.mycorrhiza_infection_probability * struct_mass_fungus * length * self.time_step)
            
            if (np.random.random() < local_infection_probability and mycorrhiza_infected_length + (2 * self.mycorrhiza_internal_infection_speed * self.time_step) < length) or (
                mycorrhiza_infected_length > 0.):
                # If new infection occurs in segment or the segment is already infected
                mycorrhiza_infected_length += 2 * self.mycorrhiza_internal_infection_speed * self.time_step
                
                if mycorrhiza_infected_length > length:
                    infection_to_parent = (mycorrhiza_infected_length - length) / 2
                    parent_id = self.g.parent(vertex_index)
                    if parent_id:
                        parent = self.g.node(parent_id)
                        if parent.length - parent.mycorrhiza_infected_length > infection_to_parent:
                            parent.mycorrhiza_infected_length += infection_to_parent
                        else:
                            parent.mycorrhiza_infected_length = parent.length

                    children = self.g.children(vertex_index)
                    if len(children) > 0:
                        infection_to_children = infection_to_parent / len(children)
                        for child_id in children:
                            child = self.g.node(child_id)
                            if child.mycorrhiza_infected_length is None:
                                child.mycorrhiza_infected_length = 0.
                            if child.length - child.mycorrhiza_infected_length > infection_to_children:
                                child.mycorrhiza_infected_length += infection_to_children
                            else:
                                child.mycorrhiza_infected_length = child.length
                    
                    mycorrhiza_infected_length = length

        return mycorrhiza_infected_length

    @rate
    def _mycorrhizal_mediated_import_Nm(self, vertex_index, mycorrhiza_infected_length, distance_from_tip, struct_mass_fungus, length, Nm_fungus, soil_temperature):
        """
        Mainly Ammonium active export by AMF to roots as reported from 
        """

        self.props["mycorrhiza_infected_length"][vertex_index] = self._mycorrhiza_infected_length(vertex_index, mycorrhiza_infected_length, distance_from_tip, struct_mass_fungus, length)


        vmax_Nm_to_roots_fungus = self.vmax_Nm_to_roots_fungus * self.temperature_modification(soil_temperature=soil_temperature,
                                                                                            T_ref=self.active_processes_T_ref,
                                                                                            A=self.active_processes_A,
                                                                                            B=self.active_processes_B,
                                                                                            C=self.active_processes_C)
        return vmax_Nm_to_roots_fungus * self.props["mycorrhiza_infected_length"][vertex_index] * Nm_fungus / (Nm_fungus + self.Km_Nm_to_roots_fungus)


    @totalrate
    def _cytokinin_synthesis(self, total_living_struct_mass, C_hexose_average, C_Nm_average, soil_temperature):
        smax_cytok = self.smax_cytok * self.temperature_modification(soil_temperature=np.mean(list(soil_temperature.values())),
                                                                                            T_ref=self.active_processes_T_ref,
                                                                                            A=self.active_processes_A,
                                                                                            B=self.active_processes_B,
                                                                                            C=self.active_processes_C)
        C_massic_concentration = C_hexose_average[1] / 6
        Ni_massic_concentration = C_Nm_average[1]

        return total_living_struct_mass[1] * smax_cytok * (
                (C_massic_concentration ** 3) / ((C_massic_concentration ** 3) + (self.Km_C_cytok ** 3))) * (
                Ni_massic_concentration / (Ni_massic_concentration + self.Km_N_cytok))


    # @note CONCENTRATIONS UPDATE

    @state
    # UPDATE NITROGEN POOLS
    def _Nm(self, Nm, living_struct_mass, import_Nm, mycorrhizal_mediated_import_Nm, diffusion_Nm_soil, diffusion_Nm_xylem, 
            export_Nm, AA_synthesis, AA_catabolism, nitrogenase_fixation, deficit_Nm) -> tuple[float, str, float]:
    
        balance = Nm + (self.time_step / living_struct_mass) * (
                import_Nm
                + mycorrhizal_mediated_import_Nm
                - diffusion_Nm_soil
                + diffusion_Nm_xylem
                - export_Nm
                - AA_synthesis * self.r_Nm_AA
                + AA_catabolism / self.r_Nm_AA
                + nitrogenase_fixation
                - deficit_Nm)
            
        deficit = - balance * living_struct_mass / self.time_step
        deficit = np.where(deficit > 1e-20, deficit, 0.)
        balance = np.maximum(balance, 0.)

        return balance, 'deficit_Nm', deficit


    @state
    def _AA(self, AA, living_struct_mass, diffusion_AA_phloem, unloading_AA_phloem, import_AA, diffusion_AA_soil, export_AA, AA_synthesis,
                  hexose_consumption_by_growth, storage_synthesis, storage_catabolism, AA_catabolism, deficit_AA) -> tuple[float, str, float]:
        
        balance =  AA + (self.time_step / living_struct_mass) * (
                diffusion_AA_phloem
                + unloading_AA_phloem
                + import_AA
                - diffusion_AA_soil
                - export_AA
                + AA_synthesis
                - (hexose_consumption_by_growth * 6 * 12 / 0.44) * self.struct_mass_N_content / self.r_Nm_AA # replaces amino_acids_consumption_by_growth
                - storage_synthesis * self.r_AA_stor
                + storage_catabolism / self.r_AA_stor
                - AA_catabolism
                - deficit_AA)

        deficit = - balance * living_struct_mass / self.time_step
        deficit = np.where(deficit > 1e-20, deficit, 0.)
        balance = np.maximum(balance, 0.)

        return balance, 'deficit_AA', deficit


    @state
    def _storage_protein(self, storage_protein, living_struct_mass, storage_synthesis, storage_catabolism):
        return storage_protein + (self.time_step / living_struct_mass) * (
                storage_synthesis
                - storage_catabolism)


    @state
    def _C_solutes_xylem(self, xylem_Nm, xylem_AA):
        N_in_total_ions=0.5
        return xylem_Nm / N_in_total_ions + xylem_AA


    @state
    def _C_solutes_phloem(self, C_sucrose_root, phloem_AA):
        ions_proportion = 0.4 # To account for high 300 mM concentrations of potassium in phloem sap, related to sucrose symport co-transport Diant et al. 2010
        return (phloem_AA) / (1 - ions_proportion) # TODO : Sucrose was removed here because the current unloading created crazy concentrations, needs to be coupled later
    

    # @note PLANT SCALE PROPERTIES UPDATE
    
    @totalstate
    def _total_cytokinins(self, total_cytokinins, cytokinin_synthesis, cytokinins_root_shoot_xylem):
        return total_cytokinins[1] + cytokinin_synthesis[1] * self.time_step - cytokinins_root_shoot_xylem[1]

    @totalstate
    def _C_Nm_average(self, Nm, living_struct_mass, total_living_struct_mass):
        return sum([x*y for x, y in zip(Nm.values(), living_struct_mass.values())]) / total_living_struct_mass[1]

    @totalstate
    def _C_AA_average(self, AA, living_struct_mass, total_living_struct_mass):
        return sum([x * y for x, y in zip(AA.values(), living_struct_mass.values())]) / total_living_struct_mass[1]

    @totalstate
    def _C_xylem_Nm_average(self, xylem_Nm, living_struct_mass, total_living_struct_mass):
        return sum([x*y for x, y in zip(xylem_Nm.values(), living_struct_mass.values())])  / total_living_struct_mass[1]

    @totalstate
    def _C_xylem_AA_average(self, xylem_AA, living_struct_mass, total_living_struct_mass):
        return sum([x*y for x, y in zip(xylem_AA.values(), living_struct_mass.values())])  / total_living_struct_mass[1]

    @totalstate
    def _C_phloem_AA_average(self, phloem_AA, living_struct_mass, total_living_struct_mass):
        return sum([x*y for x, y in zip(phloem_AA.values(), living_struct_mass.values())])  / total_living_struct_mass[1]

    @totalstate
    def _total_AA_rhizodeposition(self, diffusion_AA_soil, import_AA):
        return self.time_step * (sum(diffusion_AA_soil.values()) - sum(import_AA.values()))

    @totalstate
    def _C_hexose_average(self, living_struct_mass, total_living_struct_mass, C_hexose_root=1e-4):
        return sum([x*y for x, y in zip(C_hexose_root.values(), living_struct_mass.values())]) / total_living_struct_mass[1]


    # DERIVATIVES COMPUTED ONLY FOR PLOTTING (commented)

    # For plotting only
    #@rate
    def _amino_acids_consumption_by_growth(self, hexose_consumption_by_growth):
        return (hexose_consumption_by_growth * 6 * 12 / 0.44) * self.struct_mass_N_content / self.r_Nm_AA 

    # For plotting only
    #@state
    def _net_mineral_N_uptake(self, import_Nm, mycorrhizal_mediated_import_Nm, diffusion_Nm_soil, apoplastic_Nm_soil_xylem):
        return import_Nm + mycorrhizal_mediated_import_Nm - diffusion_Nm_soil - apoplastic_Nm_soil_xylem

    # For plotting only
    #@state
    def _net_N_uptake(self, import_Nm, import_AA, mycorrhizal_mediated_import_Nm, diffusion_Nm_soil, diffusion_AA_soil, apoplastic_Nm_soil_xylem, apoplastic_AA_soil_xylem):
        return import_Nm + import_AA + mycorrhizal_mediated_import_Nm - diffusion_Nm_soil - diffusion_AA_soil - apoplastic_Nm_soil_xylem - apoplastic_AA_soil_xylem

