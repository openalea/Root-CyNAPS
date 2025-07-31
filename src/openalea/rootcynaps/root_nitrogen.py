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

from openalea.metafspm.component import Model, declare
from openalea.metafspm.component_factory import *

from scipy.sparse import csc_matrix, identity, linalg
from scipy.optimize import lsq_linear

debug = True

@dataclass
class RootNitrogenModel(Model):
    """
    Root nitrogen balance model of Root_CyNAPS
    """


    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

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
    type: str = declare(default="Normal_root_after_emergence", unit="", unit_comment="", description="Example segment type provided by root growth model", 
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
    Cv_AA_phloem_collar: float = declare(default=260, unit="mol.m-3", unit_comment="", description="Sucrose input rate in phloem at collar point", 
                                       min_value="", max_value="", value_comment="", references="Dinant et al. 2010", DOI="",
                                        variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    sucrose_input_rate: float = declare(default=0, unit="mol.s-1", unit_comment="", description="Sucrose input rate in phloem at collar point", 
                                       min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    cytokinins_root_shoot_xylem: float = declare(default=0, unit="mol.h-1", unit_comment="of cytokinins", description="",
                                                 min_value="", max_value="", value_comment="", references="", DOI="",
                                                 variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")

    # --- INITIALIZE MODEL STATE VARIABLES ---

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
    xylem_Nm: float =           declare(default=1e-4 / 10, unit="mol.g-1", unit_comment="of structural nitrates", description="", 
                                        min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="massic_concentration", edit_by="user")
    xylem_AA: float =           declare(default=1e-4, unit="mol.g-1", unit_comment="of amino acids", description="", 
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
    total_cytokinins: float =           declare(default=250, unit="UA", unit_comment="of cytokinins", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
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

    
    # --- INITIALIZES MODEL PARAMETERS ---

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
    

    def __init__(self, g, time_step, **scenario) -> None:

        """
        Description
        Initialization of nitrogen-related variables

        Parameters
        :param g: MTG
        """
        self.g = g
        self.props = self.g.properties()
        self.time_step = time_step
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step, data=self.props)
        self.vertices = self.g.vertices(scale=self.g.max_scale())

        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)
        self.link_self_to_mtg()

    
    @stepinit
    def initialize_cumulative(self):
        # Reinitialize for the sum of the next loop
        self.props["Nm_root_to_shoot_xylem"][1] = 0
        self.props["AA_root_to_shoot_xylem"][1] = 0
        self.props["AA_root_to_shoot_phloem"][1] = 0
        for vid in self.vertices:
            n = self.g.node(vid)
            # Cumulative flows are reinitialized
            n.cumulated_radial_exchanges_Nm_xylem = 0
            n.cumulated_radial_exchanges_AA_xylem = 0
            n.cumulated_radial_exchanges_AA_phloem = 0
            n.displaced_Nm_out_xylem = 0
            n.displaced_AA_out_xylem = 0
            n.displaced_AA_out_phloem = 0
            n.displaced_Nm_in_xylem = 0
            n.displaced_AA_in_xylem = 0
            n.displaced_AA_in_phloem = 0

    # NITROGEN PROCESSES

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
        Km_LATS_Nm_root = max(0, self.Km_LATS_Nm_decrease_slope * Nm + self.Km_LATS_Nm_origin) #: Rate constant for nitrates influx at High soil N concentration; LATS linear phase
        
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
        Km_LATS_Nm_root = max(0, self.Km_LATS_Nm_decrease_slope * Nm + self.Km_LATS_Nm_origin) #: Rate constant for nitrates influx at High soil N concentration; LATS linear phase
        
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
        if np.isnan(result) or np.isinf(result):
            return 0
        else:
            return max(result, 0.)
        

    @rate
    def _diffusion_Nm_soil(self, Nm, soil_Nm, root_exchange_surface, living_struct_mass, symplasmic_volume, soil_temperature):
        if symplasmic_volume <= 0:
            return 0.
        else:
            # Passive radial diffusion between soil and cortex.
            # It happens only through root segment external surface.
            # We summarize apoplasm-soil and cortex-soil diffusion in 1 flow.
            diffusion_soil = self.diffusion_soil * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.passive_processes_T_ref,
                                                                     A=self.passive_processes_A,
                                                                     B=self.passive_processes_B,
                                                                     C=self.passive_processes_C)
            return (diffusion_soil * ((Nm * living_struct_mass / symplasmic_volume) - soil_Nm) * root_exchange_surface)

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
        if xylem_volume <= 0.:
            return 0.
        else:
            if endodermis_conductance_factor != 0:
                # If water is imported from the soil
                if radial_import_water_xylem_apoplastic > 0:
                    advection_process = - soil_Nm * radial_import_water_xylem_apoplastic # Here we compure a flux leaving the segment, but here it enters

                # this is an outflow
                else:
                    advection_process = 0 # Since we don't account for apoplasm, in this situation instead of a direct outflow to soil, we expect that this would be reuptaken by the root
                    # advection_process = - (xylem_Nm * living_struct_mass / xylem_volume) * radial_import_water_xylem_apoplastic # accounts for xylem opening and endodermis conductance already
                
                # advection_process = min(0, advection_process)

                # Direct diffusion between soil and xylem when 1) xylem is apoplastic and 2) endoderm is not differentiated
                # Here, surface is not really representative of a structure as everything is apoplasmic
                diffusion_apoplasm = self.diffusion_apoplasm * self.temperature_modification(soil_temperature=soil_temperature,
                                                                        T_ref=self.passive_processes_T_ref,
                                                                        A=self.passive_processes_A,
                                                                        B=self.passive_processes_B,
                                                                        C=self.passive_processes_C)
                diffusion_process = diffusion_apoplasm * (xylem_Nm * living_struct_mass / xylem_volume - soil_Nm) * 2 * np.pi * radius * length * xylem_differentiation_factor * endodermis_conductance_factor
                # print(advection_process, diffusion_process)
                return advection_process + diffusion_process

            else:
                return 0.


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
        if symplasmic_volume <= 0:
            return 0.
        else:
            # We define amino acid passive diffusion to soil
            diffusion_soil = self.diffusion_soil * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.passive_processes_T_ref,
                                                                     A=self.passive_processes_A,
                                                                     B=self.passive_processes_B,
                                                                     C=self.passive_processes_C)
            return (diffusion_soil * ((AA * living_struct_mass / symplasmic_volume) - soil_AA) * root_exchange_surface )

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
        if xylem_volume <= 0:
            return 0.
        else:
            if endodermis_conductance_factor != 0:
                # If water is imported from the soil
                if radial_import_water_xylem_apoplastic > 0:
                    advection_process = - soil_AA * radial_import_water_xylem_apoplastic # Here we compure a flux leaving the segment, but here it enters
                    # A corrective depending on what was actively uptaken along the way was also applied
                # this is an outflow
                else:
                    advection_process = 0 # Since we don't account for apoplasm, in this situation instead of a direct outflow to soil, we expect that this would be reuptaken by the root
                    # advection_process = - (xylem_AA * living_struct_mass / xylem_volume) * radial_import_water_xylem_apoplastic # accounts for xylem opening and endodermis conductance already

                # Direct diffusion between soil and xylem when 1) xylem is apoplastic and 2) endoderm is not differentiated
                diffusion_apoplasm = self.diffusion_apoplasm * self.temperature_modification(soil_temperature=soil_temperature,
                                                                        T_ref=self.passive_processes_T_ref,
                                                                        A=self.passive_processes_A,
                                                                        B=self.passive_processes_B,
                                                                        C=self.passive_processes_C)
                diffusion_process = diffusion_apoplasm * (xylem_AA * living_struct_mass / xylem_volume - soil_AA) * 2 * np.pi * radius * length * xylem_differentiation_factor * endodermis_conductance_factor

                return advection_process + diffusion_process

            else:
                return 0.
            
            
    @rate
    def _diffusion_AA_phloem(self, hexose_consumption_by_growth, AA, phloem_AA, phloem_exchange_surface, soil_temperature, living_struct_mass, symplasmic_volume, phloem_volume):
        """ Passive radial diffusion between phloem and cortex through plasmodesmata """
        Cv_AA_phloem = (phloem_AA * living_struct_mass) / phloem_volume
        
        if Cv_AA_phloem <= (AA * living_struct_mass) / symplasmic_volume / 2.:
                return 0
    
        else:

            AA_consumption_by_growth = (hexose_consumption_by_growth * 6 * 12 / 0.44) * self.struct_mass_N_content / self.r_Nm_AA

            diffusion_phloem = self.diffusion_phloem * (1 + (AA_consumption_by_growth / living_struct_mass) / self.reference_rate_of_AA_consumption_by_growth)

            diffusion_phloem *= self.temperature_modification(soil_temperature=soil_temperature,
                                                                        T_ref=self.passive_processes_T_ref,
                                                                        A=self.passive_processes_A,
                                                                        B=self.passive_processes_B,
                                                                        C=self.passive_processes_C)

            return diffusion_phloem * (max(0, (phloem_AA * living_struct_mass) / phloem_volume) - max(0, (AA * living_struct_mass) / symplasmic_volume)) * phloem_exchange_surface
        

    @rate
    def _unloading_AA_phloem(self, AA, phloem_AA, hexose_consumption_by_growth, phloem_exchange_surface, soil_temperature, living_struct_mass, phloem_volume, symplasmic_volume):
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
        
        return min(vmax_unloading_AA_phloem * Cv_AA_phloem * phloem_exchange_surface / (
                    self.km_unloading_AA_phloem + phloem_AA), phloem_AA * living_struct_mass / 2)


    @axial
    @rate
    def _axial_transport_N(self):
        """
        Transient resolution of solute advection
        """
        # Initialize vectors that will be incremented
        living_struct_mass = []

        row_xylem = []
        col_xylem = []
        data_xylem = []
        xylem_volume = []
        
        row_phloem = []
        col_phloem = []
        data_phloem = []
        phloem_volume = []

        radial_Nm_influx_xylem = []
        radial_AA_influx_xylem = []
        Cv_Nm_xylem = []
        Cv_AA_xylem = []
        
        radial_AA_influx_phloem = []
        Cv_AA_phloem = []

        g = self.g
        props = g.properties()
        struct_mass = g.property('struct_mass')

        local_vid = 0
        local_vids = {}
        for vid, value in struct_mass.items():
            if value > 0:
                local_vids[vid] = local_vid
                local_vid += 1

        elt_number = len(local_vids)
        boundary_Nm_xylem = np.zeros(elt_number)
        boundary_AA_xylem = np.zeros(elt_number)
        boundary_AA_phloem = np.zeros(elt_number)

        for v in self.vertices:
            n = g.node(v)
            if n.struct_mass > 0:
                Cv_Nm_xylem.append(n.xylem_Nm * n.living_struct_mass / n.xylem_volume)
                Cv_AA_xylem.append(n.xylem_AA * n.living_struct_mass / n.xylem_volume)
                Cv_AA_phloem.append(n.phloem_AA * n.living_struct_mass / n.phloem_volume)
                xylem_volume.append(n.xylem_volume)
                phloem_volume.append(n.phloem_volume)
                living_struct_mass.append(n.living_struct_mass)

                radial_Nm_influx_xylem.append(n.export_Nm - n.apoplastic_Nm_soil_xylem - n.diffusion_Nm_xylem)
                radial_AA_influx_xylem.append(n.export_AA - n.apoplastic_AA_soil_xylem)
                radial_AA_influx_phloem.append(- n.diffusion_AA_phloem - n.unloading_AA_phloem)

                # First build xylem matrix
                xylem_axial_outflow_current = 0

                # Outflux goes to the diagonal
                if n.axial_export_water_up_xylem > 0:
                    xylem_axial_outflow_current += n.axial_export_water_up_xylem   
                # Influx goes to parent column since it will multiply parent concentration             
                else:
                    if v in self.collar_children:
                        parent = 1
                        row_xylem.append(local_vids[v])
                        col_xylem.append(local_vids[parent])
                        data_xylem.append(-n.axial_export_water_up_xylem)
                    else:
                        parent = g.parent(v)
                        # If we pull from collar, we apply a Dirichet boundary condition
                        if parent is None:
                            boundary_Nm_xylem[0] = - n.axial_export_water_up_xylem * props["Cv_Nm_xylem_collar"][1]
                            boundary_AA_xylem[0] = - n.axial_export_water_up_xylem * props["Cv_AA_xylem_collar"][1]
                        else:
                            row_xylem.append(local_vids[v])
                            col_xylem.append(local_vids[parent])
                            data_xylem.append(-n.axial_export_water_up_xylem)

                if v == 1:
                    children = self.collar_children
                else:
                    children = g.children(v)

                for cid in children:
                    cn = g.node(cid)
                    # Influx goes to child column since it will multiply child concentration
                    if cn.axial_export_water_up_xylem > 0:
                        row_xylem.append(local_vids[v])
                        col_xylem.append(local_vids[cid])
                        data_xylem.append(cn.axial_export_water_up_xylem)
                    # Outflux goes to the diagonal
                    else:
                        xylem_axial_outflow_current += (- cn.axial_export_water_up_xylem)

                if xylem_axial_outflow_current > 0:
                    row_xylem.append(local_vids[v])
                    col_xylem.append(local_vids[v])
                    data_xylem.append(- xylem_axial_outflow_current)


                # Second build phloem matrix
                phloem_axial_outflow_current = 0

                # Outflux goes to the diagonal
                if n.axial_export_water_up_phloem > 0:
                    phloem_axial_outflow_current += n.axial_export_water_up_phloem
                # Influx goes to parent column since it will multiply parent concentration             
                else:
                    if v in self.collar_children:
                        parent = 1
                        row_phloem.append(local_vids[v])
                        col_phloem.append(local_vids[parent])
                        data_phloem.append(- n.axial_export_water_up_phloem)
                    else:
                        parent = g.parent(v)
                        # If we pull from collar, we apply a Dirichet boundary condition
                        if parent is None:
                            boundary_AA_phloem[0] = - n.axial_export_water_up_phloem * props["Cv_AA_phloem_collar"][1]
                        else:
                            row_phloem.append(local_vids[v])
                            col_phloem.append(local_vids[parent])
                            data_phloem.append(-n.axial_export_water_up_phloem)

                if v == 1:
                    children = self.collar_children
                else:
                    children = g.children(v)

                for cid in children:
                    cn = g.node(cid)
                    # Influx goes to child column since it will multiply child concentration
                    if cn.axial_export_water_up_phloem > 0:
                        row_phloem.append(local_vids[v])
                        col_phloem.append(local_vids[cid])
                        data_phloem.append(cn.axial_export_water_up_phloem)
                    # Outflux goes to the diagonal
                    else:
                        phloem_axial_outflow_current += (- cn.axial_export_water_up_phloem)

                if phloem_axial_outflow_current > 0:
                    row_phloem.append(local_vids[v])
                    col_phloem.append(local_vids[v])
                    data_phloem.append(- phloem_axial_outflow_current)
        
        # Xylem
        # Static components
        A_xylem = csc_matrix((data_xylem, (row_xylem, col_xylem)), shape = (elt_number, elt_number))
        xylem_volume = np.array(xylem_volume)
        living_struct_mass = np.array(living_struct_mass)
        R_Nm_xylem = np.array(radial_Nm_influx_xylem)
        R_AA_xylem = np.array(radial_AA_influx_xylem)

        # Initial conditions
        Cv_Nm_xylem = np.array(Cv_Nm_xylem)
        Cv_AA_xylem = np.array(Cv_AA_xylem)

        # Identity matrix (I)
        I = identity(elt_number, format="csc")

        # LHS = I - dt * (V^{-1} A)
        LHS_xylem = I - self.time_step * (A_xylem.multiply(1.0 / xylem_volume[:, None]))

        # RHS = C^n + dt * V^{-1} * (R + boundary)
        RHS_Nm_xylem = Cv_Nm_xylem + self.time_step * (R_Nm_xylem + boundary_Nm_xylem) / xylem_volume

        # RHS = C^n + dt * V^{-1} * (R + boundary)
        RHS_AA_xylem = Cv_AA_xylem + self.time_step * (R_AA_xylem + boundary_AA_xylem) / xylem_volume

        # Phloem
        # Static components
        A_phloem = csc_matrix((data_phloem, (row_phloem, col_phloem)), shape = (elt_number, elt_number))
        phloem_volume = np.array(phloem_volume)
        R_AA_phloem = np.array(radial_AA_influx_phloem)

        # Initial conditions
        Cv_AA_phloem = np.array(Cv_AA_phloem)

        # LHS = I - dt * (V^{-1} A)
        LHS_phloem = I - self.time_step * (A_phloem.multiply(1.0 / phloem_volume[:, None]))

        # RHS = C^n + dt * V^{-1} * (R + boundary)
        RHS_AA_phloem = Cv_AA_phloem + self.time_step * (R_AA_phloem + boundary_AA_phloem) / phloem_volume

        # Solve for C^{n+1}
        res_Nm_xylem = lsq_linear(
            A=LHS_xylem,
            b=RHS_Nm_xylem,
            bounds=(0, 1e3),
            tol=1e-3
        )
        Cv_Nm_xylem_sol = res_Nm_xylem.x
        # Cv_Nm_xylem_sol = linalg.spsolve(LHS_xylem, RHS_Nm_xylem) # Was too unstable
        assert not np.any(Cv_Nm_xylem_sol < 0)

        xylem_Nm_sol = Cv_Nm_xylem_sol * xylem_volume / living_struct_mass
        props['xylem_Nm'].update(dict(zip(local_vids.keys(), xylem_Nm_sol)))


        # Solve xylem AA
        res_AA_xylem = lsq_linear(
            A=LHS_xylem,
            b=RHS_AA_xylem,
            bounds=(0, 1e3),
            tol=1e-3
        )
        Cv_AA_xylem_sol = res_AA_xylem.x
        # Cv_AA_xylem_sol = linalg.spsolve(LHS_xylem, RHS_AA_xylem) # Was too unstable
        assert not np.any(Cv_AA_xylem_sol < 0)
        
        xylem_AA_sol = Cv_AA_xylem_sol * xylem_volume / living_struct_mass
        props['xylem_AA'].update(dict(zip(local_vids.keys(), xylem_AA_sol)))


        # Solve phloem AA
        res_AA_phloem = lsq_linear(
            A=LHS_phloem,
            b=RHS_AA_phloem,
            bounds=(0, 1e4),
            tol=1e-3
        )
        Cv_AA_phloem_sol = res_AA_phloem.x
        # Cv_AA_phloem_sol = linalg.spsolve(LHS_phloem, RHS_AA_phloem)
        assert not np.any(Cv_AA_phloem_sol < 0)

        phloem_AA_sol = Cv_AA_phloem_sol * phloem_volume / living_struct_mass
        props['phloem_AA'].update(dict(zip(local_vids.keys(), phloem_AA_sol)))


        # print("xylem Nm", Cv_Nm_xylem_sol)
        # print("xylem AA", Cv_AA_xylem_sol)
        # print("phloem AA", Cv_AA_phloem_sol)

        # If the collar flux has not been assigned by a downward flux yet, we have to compute the outflux to shoot from system balance
        # Deducted from equation C_{t+1}.V - (C_t.V + R.\Delta t - outflux.\Delta t) = 0
        props["Nm_root_to_shoot_xylem"][1] = (R_Nm_xylem * self.time_step + xylem_volume * (Cv_Nm_xylem - Cv_Nm_xylem_sol)).sum()
        props["AA_root_to_shoot_xylem"][1] = (R_AA_xylem * self.time_step + xylem_volume * (Cv_AA_xylem - Cv_AA_xylem_sol)).sum()
        props["AA_root_to_shoot_phloem"][1] = (R_AA_phloem * self.time_step + phloem_volume * (Cv_AA_phloem - Cv_AA_phloem_sol)).sum()


    # METABOLIC PROCESSES
    @rate
    def _AA_synthesis(self, living_struct_mass, Nm, soil_temperature, C_hexose_root=1e-4):
        # amino acid synthesis
        if C_hexose_root > 0 and Nm > 0:
            smax_AA = self.smax_AA * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)
            return living_struct_mass * smax_AA / (
                    ((1 + self.Km_Nm_AA) / Nm) + ((1 + self.Km_C_AA) / C_hexose_root))
        else:
            return 0
        
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

    # NOTE only for outputs
    @rate
    def _amino_acids_consumption_by_growth(self, hexose_consumption_by_growth):
        return (hexose_consumption_by_growth * 6 * 12 / 0.44) * self.struct_mass_N_content / self.r_Nm_AA 

    @rate
    def _nitrogenase_fixation(self, type, living_struct_mass, C_hexose_root, Nm, soil_temperature):
        if type == "Root_nodule":
            # We model nitrogenase expression repression by higher nitrogen availability through an inibition law
            vmax_bnf = (self.vmax_bnf / (1 + (Nm / self.K_bnf_Nm_inibition))) * self.temperature_modification(soil_temperature=soil_temperature,
                                                                                                            T_ref=self.active_processes_T_ref,
                                                                                                            A=self.active_processes_A,
                                                                                                            B=self.active_processes_B,
                                                                                                            C=self.active_processes_C)
            # Michaelis-Menten formalism
            return living_struct_mass * vmax_bnf * C_hexose_root / (self.Km_hexose_bnf + C_hexose_root)
        else:
            return 0.
        
    @state
    def _mycorrhiza_infected_length(self, vertex_index, mycorrhiza_infected_length, distance_from_tip, struct_mass_fungus, length):
        """
        From Scnepf et al 2016, modified with distance from tip here to avoid not computed root age
        NOTE : When coupling with a carbon model, mycorrhiza_infection_probability could be plasticized by mycorrhiza and C signaling exudation
        NOTE : When coupling with a carbon model, mycorrhiza_internal_infection_speed could be plasticized by C allocation to mycorrhiza.
        """

        if mycorrhiza_infected_length < length:
            # If a progress of infection is still possible
            local_infection_probability = max(1 - distance_from_tip / self.mycorrhiza_max_distance_from_tip, 0.) * (
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
        return total_living_struct_mass[1] * smax_cytok * (
                C_hexose_average[1] / (C_hexose_average[1] + self.Km_C_cytok)) * (
                C_Nm_average[1] / (C_Nm_average[1] + self.Km_N_cytok))

    
    @totalrate
    def _simple_import_Nm(self, radius, length, soil_Nm):
        mean_soil_Nm = np.mean(list(soil_Nm.values()))
        total_root_exchange_surface = sum([2 * np.pi * r * l for r, l in zip(radius.values(), length.values())])
        vmax_Nm_root = 1e-6
        Km_Nm_root_HATS = 1e-3
        return (vmax_Nm_root * mean_soil_Nm / (Km_Nm_root_HATS + mean_soil_Nm)) * total_root_exchange_surface
    

    @state
    # UPDATE NITROGEN POOLS
    def _Nm(self, vertex_index, Nm, living_struct_mass, import_Nm, mycorrhizal_mediated_import_Nm, diffusion_Nm_soil, diffusion_Nm_xylem, export_Nm, AA_synthesis, AA_catabolism, nitrogenase_fixation, deficit_Nm):
        if living_struct_mass > 0:
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
            
            # if debug: print(vertex_index, Nm, living_struct_mass, import_Nm, mycorrhizal_mediated_import_Nm, diffusion_Nm_soil, diffusion_Nm_xylem, export_Nm, AA_synthesis, AA_catabolism, nitrogenase_fixation, deficit_Nm)
            
            if balance < 0.:
                if debug: print("Deficit Nm for", vertex_index)
                if debug: print(', '.join(f"{k}: {v}" for k, v in locals().items() if k != 'self'))
                deficit = - balance * (living_struct_mass) / self.time_step
                self.props["deficit_Nm"][vertex_index] = deficit if deficit > 1e-20 else 0.
                return 0.
            else:
                self.props["deficit_Nm"][vertex_index] = 0.
                return balance
        else:
            return 0


    @state
    def _AA(self, vertex_index, AA, living_struct_mass, diffusion_AA_phloem, unloading_AA_phloem, import_AA, diffusion_AA_soil, export_AA, AA_synthesis,
                  hexose_consumption_by_growth, storage_synthesis, storage_catabolism, AA_catabolism, deficit_AA):
        
        if living_struct_mass > 0:
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
            
            # if hexose_consumption_by_growth > 0:
            #     if debug: print(', '.join(f"{k}: {v}" for k, v in locals().items() if k != 'self'))

            if balance < 0.:
                if debug: print("Deficit AA for", vertex_index)
                if debug: print(', '.join(f"{k}: {v}" for k, v in locals().items() if k != 'self'))
                deficit = - balance * (living_struct_mass) / self.time_step
                self.props["deficit_AA"][vertex_index] = deficit if deficit > 1e-20 else 0.
                return 0.
            else:
                self.props["deficit_AA"][vertex_index] = 0.
                return balance

        else:
            return 0

    @state
    def _storage_protein(self, storage_protein, living_struct_mass, storage_synthesis, storage_catabolism):
        if living_struct_mass > 0:
            return storage_protein + (self.time_step / living_struct_mass) * (
                    storage_synthesis
                    - storage_catabolism
            )
        else:
            return 0

    # @state
    def _xylem_Nm(self, vertex_index, xylem_Nm, displaced_Nm_in_xylem, displaced_Nm_out_xylem, cumulated_radial_exchanges_Nm_xylem, deficit_Nm_xylem, living_struct_mass):
        if living_struct_mass > 0:
            # Vessel's nitrogen pool update
            # Xylem balance accounting for exports from all neighbors accessible by water flow
            balance = xylem_Nm + (displaced_Nm_in_xylem 
                                  - displaced_Nm_out_xylem 
                                  + cumulated_radial_exchanges_Nm_xylem
                                  - deficit_Nm_xylem
                                  ) / living_struct_mass
            
            # if debug: print(vertex_index, xylem_Nm, displaced_Nm_in_xylem, displaced_Nm_out_xylem, cumulated_radial_exchanges_Nm_xylem, deficit_Nm_xylem, living_struct_mass)
            
            if balance < 0.:
                if debug: print("xylem Nm deficit for", vertex_index)
                deficit = - balance * (living_struct_mass) / self.time_step
                # if debug: print(', '.join(f"{k}: {v}" for k, v in locals().items() if k != 'self'))
                self.props["deficit_Nm_xylem"][vertex_index] = deficit if deficit > 1e-20 else 0.
                return 0.
            else:
                self.props["deficit_Nm_xylem"][vertex_index] = 0.
                return balance
        else:
            return 0.

    # @state
    def _xylem_AA(self, vertex_index, xylem_AA, displaced_AA_in_xylem, displaced_AA_out_xylem, cumulated_radial_exchanges_AA_xylem, deficit_AA_xylem, living_struct_mass):
        if living_struct_mass > 0:
            balance = xylem_AA + (displaced_AA_in_xylem 
                                  - displaced_AA_out_xylem 
                                  + cumulated_radial_exchanges_AA_xylem 
                                  - deficit_AA_xylem) / living_struct_mass

            if balance < 0.:
                if debug: print("xylem AA deficit for", vertex_index)
                # if debug: print(', '.join(f"{k}: {v}" for k, v in locals().items() if k != 'self'))
                deficit = - balance * (living_struct_mass) / self.time_step
                self.props["deficit_AA_xylem"][vertex_index] = deficit if deficit > 1e-20 else 0.
                return 0.
            else:
                self.props["deficit_AA_xylem"][vertex_index] = 0.
                return balance
        else:
            return 0
        
    # @state
    def _phloem_AA(self, vertex_index, type, label, phloem_AA, displaced_AA_in_phloem, displaced_AA_out_phloem, cumulated_radial_exchanges_AA_phloem, deficit_AA_phloem, living_struct_mass):
        if living_struct_mass > 0:
            balance = phloem_AA + (displaced_AA_in_phloem 
                                  - displaced_AA_out_phloem 
                                  + cumulated_radial_exchanges_AA_phloem 
                                  - deficit_AA_phloem) / living_struct_mass

            if balance < 0.:
                if debug: print("phloem AA deficit for", vertex_index)
                if debug: print(', '.join(f"{k}: {v}" for k, v in locals().items() if k != 'self'))
                deficit = - balance * (living_struct_mass) / self.time_step
                self.props["deficit_AA_phloem"][vertex_index] = deficit if deficit > 1e-20 else 0.
                return 0.
            else:
                self.props["deficit_AA_phloem"][vertex_index] = 0.
                return balance
        else:
            return 0

    @segmentation
    @state
    def _C_solutes_xylem(self, xylem_Nm, xylem_AA):
        N_in_total_ions=0.5
        return xylem_Nm / N_in_total_ions + xylem_AA
    

    @segmentation
    @state
    def _C_solutes_phloem(self, C_sucrose_root, phloem_AA):
        ions_proportion = 0.4 # To account for high 300 mM concentrations of potassium in phloem sap, related to sucrose symport co-transport Diant et al. 2010
        return (phloem_AA) / (1 - ions_proportion) # TODO : Sucrose was removed here because the current unloading created crazy concentrations, needs to be coupled later
    
    
    # For plotting only
    @state
    def _net_mineral_N_uptake(self, import_Nm, mycorrhizal_mediated_import_Nm, diffusion_Nm_soil, apoplastic_Nm_soil_xylem):
        return import_Nm + mycorrhizal_mediated_import_Nm - diffusion_Nm_soil - apoplastic_Nm_soil_xylem

    # For plotting only
    @state
    def _net_N_uptake(self, import_Nm, import_AA, mycorrhizal_mediated_import_Nm, diffusion_Nm_soil, diffusion_AA_soil, apoplastic_Nm_soil_xylem, apoplastic_AA_soil_xylem):
        return import_Nm + import_AA + mycorrhizal_mediated_import_Nm - diffusion_Nm_soil - diffusion_AA_soil - apoplastic_Nm_soil_xylem - apoplastic_AA_soil_xylem


    # PLANT SCALE PROPERTIES UPDATE

    # @totalstate
    def _C_phloem_AA(self, total_phloem_AA, C_phloem_AA, total_phloem_volume, diffusion_AA_phloem, unloading_AA_phloem, AA_root_to_shoot_phloem, sucrose_input_rate, deficit_AA_phloem):
        # Initialization step
        if total_phloem_AA[1] < 0:
            self.props["total_phloem_AA"][1] = C_phloem_AA[1] * total_phloem_volume[1]

        # print("OPTION", AA_root_to_shoot_phloem[1], sucrose_input_rate[1])
        if AA_root_to_shoot_phloem[1] is not None:
            balance = total_phloem_AA[1] + self.time_step * (-AA_root_to_shoot_phloem[1]
                                                            - sum(diffusion_AA_phloem.values())
                                                            - sum(unloading_AA_phloem.values())) - deficit_AA_phloem[1]
        else:
            balance = total_phloem_AA[1] + self.time_step * (sucrose_input_rate[1] * 0.25 # Winter 1992 replaced 0.74 from Hayashi et Chino 1986 measured 1.07 * stoechiometry
                                                            - sum(diffusion_AA_phloem.values())
                                                            - sum(unloading_AA_phloem.values())) - deficit_AA_phloem[1]
    
        if balance < 0.:
            self.props["deficit_AA_phloem"][1] = - balance if balance < -1e-20 else 0.
            self.props["total_phloem_AA"][1] = 0
            return 0.
        else:
            self.props["deficit_AA_phloem"][1] = 0
            self.props["total_phloem_AA"][1] = balance
            return balance / total_phloem_volume[1]

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
