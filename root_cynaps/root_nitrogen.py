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

from metafspm.component import Model, declare
from metafspm.component_factory import *


family = "N_metabolic"


@dataclass
class RootNitrogenModel(Model):
    """
    Root nitrogen balance model of Root_CyNAPS
    """

    family = family

    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

    # FROM CARBON MODEL
    C_hexose_root: float = declare(default=1e-4, unit="mol.g-1", unit_comment="of labile hexose", description="Hexose concentration in root",
                                  min_value="", max_value="", value_comment="", references="", DOI="",
                                   variable_type="input", by="model_carbon", state_variable_type="intensive", edit_by="user")
    

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
                            min_value="", max_value="", variable_type="input", by="model_temperature", state_variable_type="intensive", edit_by="user")

    # FROM ANATOMY MODEL
    root_exchange_surface: float = declare(default=0, unit="m2", unit_comment="of cell membrane", description="",
                                           min_value="", max_value="", value_comment="", references="",  DOI="", 
                                           variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    cortex_exchange_surface: float = declare(default=0, unit="m2", unit_comment="of cell membrane", description="Surface of cortex, taken as input to compute stele exchange surface by root_exchange_surface - cortex_exchange_surface", 
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
    total_phloem_volume: float = declare(default=0., unit="m3", unit_comment="", description="", 
                                  min_value="", max_value="", value_comment="", references="", DOI="",
                                  variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
                                            

    # FROM WATER BALANCE MODEL
    xylem_water: float =                declare(default=0, unit="mol", unit_comment="of water", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_water", state_variable_type="", edit_by="user")
    axial_export_water_up: float =      declare(default=0, unit="mol.s-1", unit_comment="of water", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_water", state_variable_type="", edit_by="user")
    axial_import_water_down: float =    declare(default=0, unit="mol.s-1", unit_comment="of water", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_water", state_variable_type="", edit_by="user")

    # FROM GROWTH MODEL
    length: float =                     declare(default=0, unit="m", unit_comment="of root segment", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    radius: float =                     declare(default=0, unit="m", unit_comment="of root segment", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    struct_mass: float =                declare(default=0, unit="g", unit_comment="of dry weight", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    total_living_struct_mass: float =          declare(default=0, unit="g", unit_comment="of dry weight", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    initial_struct_mass: float =        declare(default=0, unit="g", unit_comment="of dry weight", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    struct_mass_produced: float =       declare(default=0, unit="g", unit_comment="of dry weight", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    root_hairs_struct_mass_produced: float = declare(default=0, unit="g", unit_comment="of dry weight", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    thermal_time_since_emergence: float = declare(default=0, unit="°C", unit_comment="", description="", 
                                                  min_value="", max_value="", value_comment="", references="", DOI="",
                                                  variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    distance_from_tip: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example distance from tip", 
                                      min_value="", max_value="", value_comment="", references="", DOI="",
                                       variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    vertex_index: int = declare(default=1, unit="mol.s-1", unit_comment="", description="Unique vertex identifier stored for ease of value access", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="extensive", edit_by="user")

    # FROM SHOOT MODEL
    AA_root_shoot_phloem: float =       declare(default=0, unit="mol.time_step-1", unit_comment="of amino acids", description="",
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
    AA: float =                 declare(default=9e-4, unit="mol.g-1", unit_comment="of amino acids", description="",
                                        min_value=1e-5, max_value=1e-2, value_comment="", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="massic_concentration", edit_by="user")
    storage_protein: float =    declare(default=0., unit="mol.g-1", unit_comment="of storage proteins", description="", 
                                        min_value="", max_value="", value_comment="0 value for wheat", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="massic_concentration", edit_by="user")
    xylem_Nm: float =           declare(default=1e-4, unit="mol.g-1", unit_comment="of structural nitrates", description="", 
                                        min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="massic_concentration", edit_by="user")
    xylem_AA: float =           declare(default=1e-4, unit="mol.g-1", unit_comment="of amino acids", description="", 
                                        min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="input", by="model_nitrogen", state_variable_type="massic_concentration", edit_by="user")
    
    # Transport processes
    import_Nm: float =                      declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                    min_value=1e-11, max_value=1e-9, value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    nitrate_transporters_affinity_factor: float = declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="nitrate_transporters_affinity_factor, introduced to account for NRT1 signalling function when going through LATS regime", 
                                                    min_value="", max_value="", value_comment="", references="Remans et al 2006", DOI="", 
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    import_AA: float =                      declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    export_Nm: float =                      declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    export_AA: float =                      declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    diffusion_Nm_soil: float =              declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    diffusion_Nm_xylem: float =             declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="",
                                                    min_value="", max_value="", value_comment="", references="", DOI="", 
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    diffusion_Nm_soil_xylem: float =        declare(default=0., unit="mol.s-1", unit_comment="of nitrates", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    diffusion_AA_soil: float =              declare(default=0., unit="mol.s-1", unit_comment="of amino acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    diffusion_AA_phloem: float =            declare(default=0., unit="mol.s-1", unit_comment="of amino acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    diffusion_AA_soil_xylem: float =        declare(default=0., unit="mol.s-1", unit_comment="of amino acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    
    # Metabolic processes
    AA_synthesis: float =                   declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    struct_synthesis: float =               declare(default=0., unit="mol.s-1", unit_comment="of functional structure", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    storage_synthesis: float =              declare(default=0., unit="mol.s-1", unit_comment="of storage", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    AA_catabolism: float =                  declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    storage_catabolism: float =             declare(default=0., unit="mol.s-1", unit_comment="of storage", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    displaced_Nm_in: float =                declare(default=0., unit="mol.time_step-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    displaced_Nm_out: float =               declare(default=0., unit="mol.time_step-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    Nm_differential_by_water_transport: float =    declare(default=0., unit="mol.time_step-1", unit_comment="of nitrates", description="", 
                                                    min_value=-1e9, max_value=1e9, value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    displaced_AA_in: float =                declare(default=0., unit="mol.time_step-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    displaced_AA_out: float =               declare(default=0., unit="mol.time_step-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    cumulated_radial_exchanges_Nm: float =  declare(default=0., unit="mol.time_step-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    cumulated_radial_exchanges_AA: float =  declare(default=0., unit="mol.time_step-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")

    # Symbiotic-specific nitrogen exchanges
    nitrogenase_fixation: float =                  declare(default=0., unit="mol.s-1", unit_comment="of amonium", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")
    
    mycorrhiza_infected_length: float =          declare(default=0., unit="m", unit_comment="of root segment infected by mycorrhiza", description="Length of the root segment infected by AMF", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")

    mycorrhizal_mediated_import_Nm: float =          declare(default=0., unit="mol.s-1", unit_comment="of amonium", description="Transfer of inorganic nitrogen from michoriza to root", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="self_rate_state", edit_by="user")

    # Deficits
    deficit_Nm: float = declare(default=0., unit="mol.s-1", unit_comment="of mineral nitrogen", description="Mineral nitrogen deficit rate in root", 
                                         min_value="", max_value="", value_comment="", references="Hypothesis of no initial deficit", DOI="",
                                          variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    deficit_AA: float = declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="Amino acids deficit rate in root", 
                                           min_value="", max_value="", value_comment="", references="Hypothesis of no initial deficit", DOI="",
                                            variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")

    # SUMMED STATE VARIABLES

    C_Nm_average: float =                   declare(default=0., unit="mol.g-1", unit_comment="of nitrates", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    C_AA_average: float =                   declare(default=0., unit="mol.g-1", unit_comment="of amino acids", description="", 
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
    total_phloem_AA: float =            declare(default=1e-3, unit="mol", unit_comment="of amino acids", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    Nm_root_shoot_xylem: float =        declare(default=0., unit="mol.time_step-1", unit_comment="of nitrates", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    AA_root_shoot_xylem: float =        declare(default=0., unit="mol.time_step-1", unit_comment="of amino acids", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    AA_root_shoot_phloem_record: float =        declare(default=0., unit="mol.time_step-1", unit_comment="of amino acids", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    total_AA_rhizodeposition: float =   declare(default=0., unit="mol.time_step-1", unit_comment="of amino acids", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    cytokinin_synthesis: float =        declare(default=0., unit="UA.s-1", unit_comment="of cytokinin", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")

    # --- INITIALIZES MODEL PARAMETERS ---

    # time resolution
    sub_time_step: int =                declare(default=3600, unit="s", unit_comment="", description="MUST be a multiple of base time_step", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")

    # N TRANSPORT PROCESSES
    # kinetic parameters
    vmax_Nm_root: float =               declare(default=1e-6, unit="mol.s-1.m-2", unit_comment="of nitrates", description="",
                                                min_value="", max_value="", value_comment="*2 to slightly increase the impact of amino acid production", references="Liu et Tsay", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    vmax_Nm_xylem: float =              declare(default=2*1e-5, unit="mol.s-1.m-2", unit_comment="of nitrates", description="",
                                                min_value="", max_value="", value_comment="*10e2 from outside root as a lower surface has to compete with external surface and presents LATS", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_Nm_root_LATS: float =            declare(default=1e1, unit="mol.m-3", unit_comment="of nitrates", description="",
                                                min_value="", max_value="", value_comment="", references="Liu et Tsay 2003", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_Nm_root_HATS: float =            declare(default=1e-3, unit="mol.m-3", unit_comment="of nitrates", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    begin_N_regulation: float =         declare(default=1., unit="mol.g-1", unit_comment="of nitrates", description="",
                                                min_value="", max_value="", value_comment="changed so that import_Nm variation may occur in observed Nm variation range, solve boundary and middle centering equations", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    span_N_regulation: float =          declare(default=6e-5, unit="mol.g-1", unit_comment="of nitrates", description="",
                                                min_value="", max_value="", value_comment="changed so that import_Nm variation may occur in observed Nm variation range, solve boundary and middle centering equations", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_Nm_xylem: float =                declare(default=1e-1, unit="mol.g-1", unit_comment="of nitrates", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
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
    diffusion_xylem: float =            declare(default=1e-7, unit="g.s-1.m-2", unit_comment="of solute", description="",
                                                min_value="", max_value="", value_comment="from 1e-8, lowered to avoid crazy segment loading bugs", references="", DOI="", 
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    diffusion_phloem: float =           declare(default=1.2e-8, unit="g.s-1.m-2", unit_comment="of solute", description="",
                                                min_value="", max_value="", value_comment="1.2e-8 * Important value to avoid harsh growth limitations", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="I", edit_by="user")  # Artif *1e-1 g.m-2.s-1 more realistic ranges
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
    transport_C_regulation: float =     declare(default=7e-3, unit="mol.g-1", unit_comment="of hexose", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")

    # N METABOLISM PROCESSES
    # TODO : introduce nitrogen fixation
    # kinetic parameters
    smax_AA: float =                    declare(default=1e-5, unit="mol.s-1.g-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="*100 from ref to come closer to the 30% prop in whole synthesis expected", references="(Barillot 2016)", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_Nm_AA: float =                   declare(default=3.50e-6, unit="mol.g-1", unit_comment="of nitrates", description="", 
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
    cmax_AA: float =                    declare(default=5e-9, unit="mol.s-1.g-1", unit_comment="of amino acids", description="",
                                                min_value="", max_value="", value_comment="5e-9 for now not relevant as it doesn't contribute to C_hexose_root balance.", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_AA_catab: float =                declare(default=2.5e-6, unit="mol.g-1", unit_comment="of amino acids", description="", 
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
    struct_mass_N_content: float = declare(default=0.44 / 20 / 14, unit="mol.g-1", unit_comment="of carbon", description="C content of structural mass", 
                                                    min_value="", max_value="", value_comment="", references="We assume that the structural mass contains 44% of C. (??)", DOI="",
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

    def post_growth_updating(self):
        """
        Description :
            Extend property dictionary upon new element partitioning and updates concentrations upon structural_mass change
        """
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        already_updated = []
        for vid in self.vertices:
            # We ignore already updated elements, e.g. parents of apices
            if vid in already_updated:
                continue

            # If we focus on a new element
            if vid not in list(self.Nm.keys()):
                parent = self.g.parent(vid)
                for prop in self.state_variables:
                    # if intensive, equals to parent AFTER it has been updated
                    if self.__dataclass_fields__[prop].metadata["state_variable_type"] == "intensive":
                        getattr(self, prop).update({parent: getattr(self, prop)[parent] * (
                                self.initial_struct_mass[parent] / self.struct_mass[parent])})
                        getattr(self, prop).update({vid: getattr(self, prop)[parent]})
                    # if extensive, we need structural mass wise partitioning
                    else:
                        # we partition the initial flow in the parent accounting for mass fraction
                        # We use struct_mass, the resulting structural mass after growth
                        mass_fraction = self.struct_mass[vid] / (self.struct_mass[vid] + self.struct_mass[parent])
                        getattr(self, prop).update({vid: getattr(self, prop)[parent] * mass_fraction,
                                                    parent: getattr(self, prop)[parent] * (1-mass_fraction)})
                already_updated += [vid, parent]

            # If the element already exists and isn't immediate neighbor of an apex
            else:
                # If after growth the element actually grown
                if self.struct_mass[vid] > 0:
                    for prop in self.state_variables:
                        # if intensive, concentrations have to be updated based on new structural mass
                        if self.__dataclass_fields__[prop].metadata["state_variable_type"] == "intensive":
                            getattr(self, prop).update({vid: getattr(self, prop)[vid] * (
                                self.initial_struct_mass[vid] / self.struct_mass[vid])})
                        # if extensive, it doesn't need to be updated and if parent is segmented,

                already_updated += [vid]
    
    @stepinit
    def initialize_cumulative(self):
        # Reinitialize for the sum of the next loop
        self.Nm_root_shoot_xylem[1] = 0
        self.AA_root_shoot_xylem[1] = 0
        for vid in self.vertices:
            # Cumulative flows are reinitialized
            self.cumulated_radial_exchanges_Nm[vid] = 0
            self.cumulated_radial_exchanges_AA[vid] = 0
            self.displaced_Nm_out[vid] = 0
            self.displaced_AA_out[vid] = 0
            self.displaced_Nm_in[vid] = 0
            self.displaced_AA_in[vid] = 0

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
        # We define mineral nitrogen active uptake from soil
        precision = 0.99
        Km_Nm_root = (self.Km_Nm_root_LATS - self.Km_Nm_root_HATS) / (
                1 + (precision / ((1 - precision) * np.exp(-self.begin_N_regulation))
                     * np.exp(-Nm / self.span_N_regulation))
        ) + self.Km_Nm_root_HATS
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        vmax_Nm_root = self.vmax_Nm_root * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)

        return ((soil_Nm * vmax_Nm_root / (soil_Nm + Km_Nm_root)) * root_exchange_surface * (
            C_hexose_root / (C_hexose_root + self.transport_C_regulation)))
    
    @rate
    def _nitrate_transporters_affinity_factor(self, Nm):
        precision = 0.99
        # accounted for in soil heterogeneity scenario
        return 1 / (1 + (precision / ((1 - precision) * np.exp(-self.begin_N_regulation))
                     * np.exp(-Nm / self.span_N_regulation)))

    @rate
    def _diffusion_Nm_soil(self, Nm, soil_Nm, root_exchange_surface, struct_mass, symplasmic_volume, soil_temperature):
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
            return (diffusion_soil * ((Nm * struct_mass / symplasmic_volume) - soil_Nm) * root_exchange_surface)

    @rate
    def _export_Nm(self, Nm, root_exchange_surface, cortex_exchange_surface, soil_temperature, C_hexose_root=1e-4):
        # We define active export to xylem from root segment
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        vmax_Nm_xylem = self.vmax_Nm_xylem * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)
        return ((Nm * vmax_Nm_xylem) / (Nm + self.Km_Nm_xylem)) * (root_exchange_surface - cortex_exchange_surface) * (
                C_hexose_root / (C_hexose_root + self.transport_C_regulation))

    @rate
    def _diffusion_Nm_xylem(self, xylem_Nm, Nm, root_exchange_surface, cortex_exchange_surface, soil_temperature, struct_mass, symplasmic_volume, xylem_volume):
        # Passive radial diffusion between xylem and cortex through plasmalema
        diffusion_xylem = self.diffusion_xylem * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.passive_processes_T_ref,
                                                                     A=self.passive_processes_A,
                                                                     B=self.passive_processes_B,
                                                                     C=self.passive_processes_C)
        return diffusion_xylem * ((xylem_Nm * struct_mass / xylem_volume) - (Nm * struct_mass / symplasmic_volume)) * (root_exchange_surface - cortex_exchange_surface)

    @rate
    def _diffusion_Nm_soil_xylem(self, soil_Nm, xylem_Nm, radius, length, xylem_differentiation_factor, endodermis_conductance_factor, struct_mass, xylem_volume, soil_temperature):
        if xylem_volume <= 0.:
            return 0.
        else:
            # Direct diffusion between soil and xylem when 1) xylem is apoplastic and 2) endoderm is not differentiated
            # Here, surface is not really representative of a structure as everything is apoplasmic
            diffusion_apoplasm = self.diffusion_apoplasm * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.passive_processes_T_ref,
                                                                     A=self.passive_processes_A,
                                                                     B=self.passive_processes_B,
                                                                     C=self.passive_processes_C)
            return diffusion_apoplasm * (xylem_Nm * struct_mass / xylem_volume - soil_Nm) * 2 * np.pi * radius * length * xylem_differentiation_factor * endodermis_conductance_factor

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
    def _diffusion_AA_soil(self, AA, soil_AA, root_exchange_surface, struct_mass, symplasmic_volume, soil_temperature):
        if symplasmic_volume <= 0:
            return 0.
        else:
            # We define amino acid passive diffusion to soil
            diffusion_soil = self.diffusion_soil * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.passive_processes_T_ref,
                                                                     A=self.passive_processes_A,
                                                                     B=self.passive_processes_B,
                                                                     C=self.passive_processes_C)
            return (diffusion_soil * ((AA * struct_mass / symplasmic_volume) - soil_AA) * root_exchange_surface )

    @rate
    def _export_AA(self, AA, root_exchange_surface, cortex_exchange_surface, soil_temperature, C_hexose_root=1e-4):
        # We define active export to xylem from root segment
        # Km is defined as a constant here
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        vmax_AA_xylem = self.vmax_AA_xylem * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)
        return ((AA * vmax_AA_xylem / (AA + self.Km_AA_xylem))
                * (root_exchange_surface - cortex_exchange_surface) * (C_hexose_root / (
                        C_hexose_root + self.transport_C_regulation)))

    @rate
    def _diffusion_AA_soil_xylem(self, soil_AA, xylem_AA, radius, length, xylem_differentiation_factor, endodermis_conductance_factor, struct_mass, xylem_volume, soil_temperature):
        if xylem_volume <= 0:
            return 0.
        else:
            # Direct diffusion between soil and xylem when 1) xylem is apoplastic and 2) endoderm is not differentiated
            diffusion_apoplasm = self.diffusion_apoplasm * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.passive_processes_T_ref,
                                                                     A=self.passive_processes_A,
                                                                     B=self.passive_processes_B,
                                                                     C=self.passive_processes_C)
            return (diffusion_apoplasm * (xylem_AA * struct_mass / xylem_volume - soil_AA) * 2 * np.pi * radius * length * xylem_differentiation_factor * endodermis_conductance_factor)

    @rate
    def _diffusion_AA_phloem(self, AA, phloem_exchange_surface, soil_temperature, struct_mass, symplasmic_volume):
        # Passive radial diffusion between phloem and cortex through plasmodesmata
        # TODO : Change diffusive flow to enable realistic ranges, now, unloading is limited by a ping pong bug related to diffusion
        # TODO : resolve exception when mapping has to deal with plant scale properties AND local ones
        diffusion_phloem = self.diffusion_phloem * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.passive_processes_T_ref,
                                                                     A=self.passive_processes_A,
                                                                     B=self.passive_processes_B,
                                                                     C=self.passive_processes_C)
        return diffusion_phloem * (self.total_phloem_AA[1] / self.total_phloem_volume[1] - AA * struct_mass / symplasmic_volume) * phloem_exchange_surface

    @axial
    @rate
    # AXIAL TRANSPORT PROCESSES
    def _axial_transport_N(self):
        """
            Description
            ___________
        """
        # TODO : probably collar children have to be reintroduced for good neighbor management. (from model_water)
        #  But if null water content proprely passes information between collar and it's children,
        #  it may be already working well

        # AXIAL TRANSPORT
        for v in self.vertices:
            if self.struct_mass[v] > 0 :
                # If this is only an out flow to up parents
                if self.axial_export_water_up[v] * self.time_step > 0:
                    # Turnover defines a dilution factor of radial transport processes over the axially transported
                    # water column
                    turnover = self.axial_export_water_up[v] * self.time_step / self.xylem_water[v]
                    if turnover <= 1:
                        # Transport only affects considered segment
                        self.cumulated_radial_exchanges_Nm[v] += (self.export_Nm[v] - self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * self.time_step
                        self.cumulated_radial_exchanges_AA[v] += (self.export_AA[v] - self.diffusion_AA_soil_xylem[v]) * self.time_step
                        # Exported matter corresponds to the exported water proportion
                        self.displaced_Nm_out[v] = turnover * self.xylem_Nm[v] * self.struct_mass[v]
                        self.displaced_AA_out[v] = turnover * self.xylem_AA[v] * self.struct_mass[v]
                        up_parent = self.g.parent(v)
                        # If this is collar, this flow is exported
                        if up_parent == None:
                            self.Nm_root_shoot_xylem[1] += self.displaced_Nm_out[v]
                            self.AA_root_shoot_xylem[1] += self.displaced_AA_out[v]
                        else:
                            # The immediate parent receives this flow
                            self.displaced_Nm_in[up_parent] += self.displaced_Nm_out[v]
                            self.displaced_AA_in[up_parent] += self.displaced_AA_out[v]
                    else:
                        #print("Uturnover >1")
                        # Exported matter corresponds to the whole segment's water content
                        self.displaced_Nm_out[v] = self.xylem_Nm[v] * self.struct_mass[v]
                        self.displaced_AA_out[v] = self.xylem_AA[v] * self.struct_mass[v]
                        # Transport affects a chain of parents
                        water_exchange_time = self.time_step / turnover
                        # Loading of the current vertex into the current vertex's xylem
                        self.cumulated_radial_exchanges_Nm[v] += (self.export_Nm[v] - self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time
                        self.cumulated_radial_exchanges_AA[v] += (self.export_AA[v] - self.diffusion_AA_soil_xylem[v]) * water_exchange_time

                        exported_water = self.axial_export_water_up[v] * self.time_step
                        child = v
                        # Loading of the current vertex into the vertices who have received water from it
                        while exported_water > 0:
                            # We remove the amount of water which has already received loading in previous loop
                            exported_water -= self.xylem_water[child]
                            up_parent = self.g.parent(child)
                            # If we reached collar, this amount is being exported
                            if up_parent == None:
                                self.Nm_root_shoot_xylem[1] += (self.export_Nm[v] - self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * exported_water / self.xylem_water[v]
                                self.AA_root_shoot_xylem[1] += (self.export_AA[v] - self.diffusion_AA_soil_xylem[v]) * water_exchange_time * exported_water / self.xylem_water[v]
                                # If all water content of initial segment is exported through collar
                                if exported_water > self.xylem_water[v]:
                                    self.Nm_root_shoot_xylem[1] += self.displaced_Nm_out[v]
                                    self.AA_root_shoot_xylem[1] += self.displaced_AA_out[v]
                                else:
                                    parent_proportion = exported_water / self.xylem_water[v]
                                    self.Nm_root_shoot_xylem[1] += self.displaced_Nm_out[v] * parent_proportion
                                    self.AA_root_shoot_xylem[1] += self.displaced_AA_out[v] * parent_proportion
                                    self.displaced_Nm_in[child] += self.displaced_Nm_out[v] * (1 - parent_proportion)
                                    self.displaced_AA_in[child] += self.displaced_AA_out[v] * (1 - parent_proportion)
                                # Break the loop
                                exported_water = 0
                            else:
                                # If the considered parent have been completly filled with water from the child
                                if exported_water - self.xylem_water[up_parent] > 0:
                                    # The exposition time is longer if the water content of the target neighbour is more important.
                                    self.cumulated_radial_exchanges_Nm[up_parent] += (self.export_Nm[v] - self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * self.xylem_water[up_parent] / self.xylem_water[v]
                                    self.cumulated_radial_exchanges_AA[up_parent] += (self.export_AA[v] - self.diffusion_AA_soil_xylem[v]) * water_exchange_time * self.xylem_water[up_parent] / self.xylem_water[v]
                                # If it's only partial, we account only for the exceeding amount
                                else:
                                    self.cumulated_radial_exchanges_Nm[up_parent] += (self.export_Nm[v] - self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * exported_water / self.xylem_water[v]
                                    self.cumulated_radial_exchanges_AA[up_parent] += (self.export_AA[v] - self.diffusion_AA_soil_xylem[v]) * water_exchange_time * exported_water / self.xylem_water[v]
                                    # If all water content of initial segment is exported to the considered grandparent
                                    if exported_water > self.xylem_water[v]:
                                        self.displaced_Nm_in[up_parent] += self.displaced_Nm_out[v]
                                        self.displaced_AA_in[up_parent] += self.displaced_AA_out[v]
                                    else:
                                        # Displaced matter is shared between child and its parent
                                        parent_proportion = exported_water / self.xylem_water[v]
                                        self.displaced_Nm_in[up_parent] += self.displaced_Nm_out[v] * parent_proportion
                                        self.displaced_AA_in[up_parent] += self.displaced_AA_out[v] * parent_proportion
                                        self.displaced_Nm_in[child] += self.displaced_Nm_out[v] * (1 - parent_proportion)
                                        self.displaced_AA_in[child] += self.displaced_AA_out[v] * (1 - parent_proportion)
                                    # Break the loop
                                    exported_water = 0
                                child = up_parent

                # If this is only a out flow to down children
                if self.axial_import_water_down[v] * self.time_step < 0:
                    # Turnover defines a dilution factor of radial transport processes over the axially transported
                    # water column
                    turnover = - self.axial_import_water_down[v] * self.time_step / self.xylem_water[v]
                    if turnover <= 1:
                        # Transport only affects considered segment
                        self.cumulated_radial_exchanges_Nm[v] += (self.export_Nm[v] - self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * self.time_step
                        self.cumulated_radial_exchanges_AA[v] += (self.export_AA[v] - self.diffusion_AA_soil_xylem[v]) * self.time_step
                        # Exported matter corresponds to the exported water proportion
                        self.displaced_Nm_out[v] = turnover * self.xylem_Nm[v] * self.struct_mass[v]
                        self.displaced_AA_out[v] = turnover * self.xylem_AA[v] * self.struct_mass[v]
                        down_children = [k for k in self.g.children(v) if self.struct_mass[k] > 0]
                        # The immediate children receive this flow
                        radius_sum = sum([self.radius[k] for k in down_children])
                        children_radius_prop = [self.radius[k] / radius_sum for k in down_children]
                        for ch in range(len(down_children)):
                            self.displaced_Nm_in[down_children[ch]] += self.displaced_Nm_out[v] * children_radius_prop[ch]
                            self.displaced_AA_in[down_children[ch]] += self.displaced_AA_out[v] * children_radius_prop[ch]

                    else:
                        # Transport affects several segments, and we verified it often happens under high transpiration
                        # Exported matter corresponds to the whole segment's water content
                        self.displaced_Nm_out[v] = self.xylem_Nm[v] * self.struct_mass[v]
                        self.displaced_AA_out[v] = self.xylem_AA[v] * self.struct_mass[v]
                        # Transport affects a chain of children
                        water_exchange_time = self.time_step / turnover
                        # Loading of the current vertex into the current vertex's xylem
                        self.cumulated_radial_exchanges_Nm[v] += (self.export_Nm[v] - self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time
                        self.cumulated_radial_exchanges_AA[v] += (self.export_AA[v] - self.diffusion_AA_soil_xylem[v]) * water_exchange_time

                        parent = [v]
                        # We initialize a list tracking water repartition among down axes
                        axis_proportion = [1.0]
                        # We remove the amount of water which has already been received
                        exported_water = [-self.axial_import_water_down[v] * self.time_step - self.xylem_water[v]]
                        # Loading of the current vertex into the vertices who have received water from it
                        while True in [k > 0 for k in exported_water]:
                            children_list = []
                            children_exported_water = []
                            for p in range(len(parent)):
                                if exported_water[p] > 0:
                                    down_children = [k for k in self.g.children(parent[p]) if self.struct_mass[k] > 0]
                                    # if the parent is an apex and water has been exported from it,
                                    # it means that the apex concentrates the associated carried and loaded nitrogen matter
                                    if len(down_children) == 0:
                                        # this water amount has also been subject to loading
                                        self.cumulated_radial_exchanges_Nm[parent[p]] += (self.export_Nm[v] - self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * exported_water[p] / self.xylem_water[v]
                                        self.cumulated_radial_exchanges_AA[parent[p]] += (self.export_AA[v] - self.diffusion_AA_soil_xylem[v]) * water_exchange_time * exported_water[p] / self.xylem_water[v]
                                        # if the translated nitrogen matter has completely ended up in the apex
                                        if exported_water[p] + self.xylem_water[parent[p]] > self.xylem_water[v] * axis_proportion[p]:
                                            self.displaced_Nm_in[parent[p]] += self.displaced_Nm_out[v] * axis_proportion[p]
                                            self.displaced_AA_in[parent[p]] += self.displaced_AA_out[v] * axis_proportion[p]
                                        # else it is shared with grandparent
                                        else:
                                            grandparent = self.g.parent(parent[p])
                                            parent_proportion = (exported_water[p] + self.xylem_water[parent[p]]) / (self.xylem_water[v] * axis_proportion[p])
                                            self.displaced_Nm_in[parent[p]] += self.displaced_Nm_out[v] * axis_proportion[p] * parent_proportion
                                            self.displaced_AA_in[parent[p]] += self.displaced_AA_out[v] * axis_proportion[p] * parent_proportion
                                            self.displaced_Nm_in[grandparent] += self.displaced_Nm_out[v] * axis_proportion[p] * (1 - parent_proportion)
                                            self.displaced_AA_in[grandparent] += self.displaced_AA_out[v] * axis_proportion[p] * (1 - parent_proportion)

                                    # If there is only 1 child (root line)
                                    elif len(down_children) == 1:
                                        # If the considered child have been completely filled with water from the parent
                                        if exported_water[p] - self.xylem_water[down_children[0]] > 0:
                                            # The exposition time is longer if the water content of the target neighbour is more important.
                                            self.cumulated_radial_exchanges_Nm[down_children[0]] += (self.export_Nm[v] - self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * self.xylem_water[down_children[0]] / self.xylem_water[v]
                                            self.cumulated_radial_exchanges_AA[down_children[0]] += (self.export_AA[v] - self.diffusion_AA_soil_xylem[v]) * water_exchange_time * self.xylem_water[down_children[0]] / self.xylem_water[v]
                                            children_exported_water += [exported_water[p] - self.xylem_water[down_children[0]]]
                                        else:
                                            self.cumulated_radial_exchanges_Nm[down_children[0]] += (self.export_Nm[v] - self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * exported_water[p] / self.xylem_water[v]
                                            self.cumulated_radial_exchanges_AA[down_children[0]] += (self.export_AA[v] - self.diffusion_AA_soil_xylem[v]) * water_exchange_time * exported_water[p] / self.xylem_water[v]
                                            # If all water content from initial segment gone through this axis is exported to the considered child
                                            if exported_water[p] > self.xylem_water[v] * axis_proportion[p]:
                                                self.displaced_Nm_in[down_children[0]] += self.displaced_Nm_out[v] * axis_proportion[p]
                                                self.displaced_AA_in[down_children[0]] += self.displaced_AA_out[v] * axis_proportion[p]
                                            else:
                                                # Displaced matter is shared between child and its parent
                                                child_proportion = exported_water[p] / (self.xylem_water[v] * axis_proportion[p])
                                                self.displaced_Nm_in[down_children[0]] += self.displaced_Nm_out[v] * axis_proportion[p] * child_proportion
                                                self.displaced_AA_in[down_children[0]] += self.displaced_AA_out[v] * axis_proportion[p] * child_proportion
                                                self.displaced_Nm_in[parent[p]] += self.displaced_Nm_out[v] * axis_proportion[p] * (1 - child_proportion)
                                                self.displaced_AA_in[parent[p]] += self.displaced_AA_out[v] * axis_proportion[p] * (1 - child_proportion)
                                            # Break the loop
                                            children_exported_water += [0]

                                    # Else if there are several children
                                    else:
                                        # Water repartition is done according to radius,
                                        # as this is the main criteria used in the water model
                                        radius_sum = sum([self.radius[k] for k in down_children])
                                        children_down_flow = [exported_water[p] * self.radius[k] / radius_sum for k in down_children]
                                        children_radius_prop = [self.radius[k] / radius_sum for k in down_children]
                                        # Actualize the repartition of water when there is a new branching
                                        for k in range(1, len(children_radius_prop)):
                                            axis_proportion.insert(p + 1, axis_proportion[p] * children_radius_prop[-k])

                                        axis_proportion[p] = axis_proportion[p] * children_radius_prop[0]
                                        for ch in range(len(down_children)):
                                            # If the considered child have been completely filled with water from the parent
                                            if children_down_flow[ch] - self.xylem_water[down_children[ch]] > 0 :
                                                # The exposition time is longer if the water content of the target neighbour is more important.
                                                self.cumulated_radial_exchanges_Nm[down_children[ch]] += (self.export_Nm[v] - self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * self.xylem_water[down_children[ch]] / self.xylem_water[v]
                                                self.cumulated_radial_exchanges_AA[down_children[ch]] += (self.export_AA[v] - self.diffusion_AA_soil_xylem[v]) * water_exchange_time * self.xylem_water[down_children[ch]] / self.xylem_water[v]
                                                children_down_flow[ch] -= self.xylem_water[down_children[ch]]
                                            else:
                                                self.cumulated_radial_exchanges_Nm[down_children[ch]] += (self.export_Nm[v] - self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * children_down_flow[ch] / self.xylem_water[v]
                                                self.cumulated_radial_exchanges_AA[down_children[ch]] += (self.export_AA[v] - self.diffusion_AA_soil_xylem[v]) * water_exchange_time * children_down_flow[ch] / self.xylem_water[v]
                                                # If all water content from initial segment gone through this axis is exported to the considered child
                                                if children_down_flow[ch] > self.xylem_water[v] * axis_proportion[p + ch]:
                                                    self.displaced_Nm_in[down_children[ch]] += self.displaced_Nm_out[v] * axis_proportion[p + ch]
                                                    self.displaced_AA_in[down_children[ch]] += self.displaced_AA_out[v] * axis_proportion[p + ch]
                                                else:
                                                    # Displaced matter is shared between child and its parent
                                                    child_proportion = children_down_flow[ch] / (self.xylem_water[v] * axis_proportion[p + ch])
                                                    self.displaced_Nm_in[down_children[ch]] += self.displaced_Nm_out[v] * axis_proportion[p + ch] * child_proportion
                                                    self.displaced_AA_in[down_children[ch]] += self.displaced_AA_out[v] * axis_proportion[p + ch] * child_proportion
                                                    self.displaced_Nm_in[parent[p]] += self.displaced_Nm_out[v] * axis_proportion[p + ch] * (1 - child_proportion)
                                                    self.displaced_AA_in[parent[p]] += self.displaced_AA_out[v] * axis_proportion[p + ch] * (1 - child_proportion)
                                                # Break the loop
                                                children_down_flow[ch] = 0
                                        children_exported_water += children_down_flow

                                    # Concatenate so that each children will become parent for the next loop
                                    children_list += down_children

                            # Children become parent for the next loop
                            parent = children_list
                            exported_water = children_exported_water

                # If this is an inflow from both up an down segments
                if self.axial_import_water_down[v] * self.time_step >= 0 >= self.axial_export_water_up[v] * self.time_step:
                    # There is no exported matter, thus no receiver
                    self.displaced_Nm_out[v] = 0
                    self.displaced_AA_out[v] = 0
                    # No matter what the transported water amount is, all radial transport effects remain on the current vertex.
                    self.cumulated_radial_exchanges_Nm[v] += (self.export_Nm[v] - self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * self.time_step
                    self.cumulated_radial_exchanges_AA[v] += (self.export_AA[v] - self.diffusion_AA_soil_xylem[v]) * self.time_step

                self.Nm_differential_by_water_transport[v] = self.displaced_Nm_in[v] - self.displaced_Nm_out[v]

    # METABOLIC PROCESSES
    @rate
    def _AA_synthesis(self, struct_mass, Nm, soil_temperature, C_hexose_root=1e-4):
        # amino acid synthesis
        if C_hexose_root > 0 and Nm > 0:
            smax_AA = self.smax_AA * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)
            return struct_mass * smax_AA / (
                    ((1 + self.Km_Nm_AA) / Nm) + ((1 + self.Km_C_AA) / C_hexose_root))
        else:
            return 0

    @rate
    def _struct_synthesis(self, struct_mass_produced, root_hairs_struct_mass_produced):
        # Organic structure synthesis
        return (struct_mass_produced + root_hairs_struct_mass_produced) * self.struct_mass_N_content / self.r_Nm_AA
        
    @rate
    def _storage_synthesis(self, struct_mass, AA, soil_temperature):
        # Organic storage synthesis (Michaelis-Menten kinetic)
        smax_stor = self.smax_stor * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)
        return struct_mass * (smax_stor * AA / (self.Km_AA_stor + AA))

    @rate
    def _storage_catabolism(self, struct_mass, storage_protein, soil_temperature, C_hexose_root=1e-4):
        # Organic storage catabolism through proteinase
        Km_stor_root = self.Km_stor_catab * np.exp(self.storage_C_regulation * C_hexose_root)
        cmax_stor = self.cmax_stor * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)
        return struct_mass * cmax_stor * storage_protein / (Km_stor_root + storage_protein)

    @rate
    def _AA_catabolism(self, struct_mass, AA, soil_temperature, C_hexose_root=1e-4):
        # AA catabolism through GDH
        Km_stor_root = self.Km_AA_catab * np.exp(self.storage_C_regulation * C_hexose_root)
        cmax_AA = self.cmax_AA * self.temperature_modification(soil_temperature=soil_temperature,
                                                                     T_ref=self.active_processes_T_ref,
                                                                     A=self.active_processes_A,
                                                                     B=self.active_processes_B,
                                                                     C=self.active_processes_C)
        return struct_mass * cmax_AA * AA / (Km_stor_root + AA)

    #@rate
    def _nitrogenase_fixation(self, type, struct_mass, C_hexose_root, Nm, soil_temperature):
        if type == "Root_nodule":
            # We model nitrogenase expression repression by higher nitrogen availability through an inibition law
            vmax_bnf = (self.vmax_bnf / (1 + (Nm / self.K_bnf_Nm_inibition))) * self.temperature_modification(soil_temperature=soil_temperature,
                                                                                                            T_ref=self.active_processes_T_ref,
                                                                                                            A=self.active_processes_A,
                                                                                                            B=self.active_processes_B,
                                                                                                            C=self.active_processes_C)
            # Michaelis-Menten formalism
            return struct_mass * vmax_bnf * C_hexose_root / (self.Km_hexose_bnf + C_hexose_root)
        else:
            return 0.
        
    #TP@state
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
                            if child.length - child.mycorrhiza_infected_length > infection_to_children:
                                child.mycorrhiza_infected_length += infection_to_children
                            else:
                                child.mycorrhiza_infected_length = child.length
                    
                    mycorrhiza_infected_length = length

        return mycorrhiza_infected_length

    @rate
    def _mycorrhizal_mediated_import_Nm(self, mycorrhiza_infected_length, Nm_fungus, soil_temperature):
        """
        Mainly Ammonium active export by AMF to roots as reported from 
        """
        vmax_Nm_to_roots_fungus = self.vmax_Nm_to_roots_fungus * self.temperature_modification(soil_temperature=soil_temperature,
                                                                                            T_ref=self.active_processes_T_ref,
                                                                                            A=self.active_processes_A,
                                                                                            B=self.active_processes_B,
                                                                                            C=self.active_processes_C)
        return vmax_Nm_to_roots_fungus * mycorrhiza_infected_length * Nm_fungus / (Nm_fungus + self.Km_Nm_to_roots_fungus)

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

    
    @state
    # UPDATE NITROGEN POOLS
    def _Nm(self, vertex_index, Nm, struct_mass, import_Nm, mycorrhizal_mediated_import_Nm, diffusion_Nm_soil, diffusion_Nm_xylem, export_Nm, AA_synthesis, AA_catabolism, nitrogenase_fixation, deficit_Nm):
        if struct_mass > 0:
            balance = Nm + (self.time_step / struct_mass) * (
                    import_Nm
                    + mycorrhizal_mediated_import_Nm
                    - diffusion_Nm_soil
                    + diffusion_Nm_xylem
                    - export_Nm
                    - AA_synthesis * self.r_Nm_AA
                    + AA_catabolism / self.r_Nm_AA
                    + nitrogenase_fixation
                    - deficit_Nm)
            if balance < 0.:
                deficit = - balance * (struct_mass) / self.time_step
                self.deficit_Nm[vertex_index] = deficit if deficit > 1e-20 else 0.
                return 0.
            else:
                self.deficit_Nm[vertex_index] = 0.
                return balance
        else:
            return 0


    @state
    def _AA(self, vertex_index, AA, struct_mass, diffusion_AA_phloem, import_AA, diffusion_AA_soil, export_AA, AA_synthesis,
                  struct_synthesis, storage_synthesis, storage_catabolism, AA_catabolism, deficit_AA):
        
        if struct_mass > 0:
            balance =  AA + (self.time_step / struct_mass) * (
                    diffusion_AA_phloem
                    + import_AA
                    - diffusion_AA_soil
                    - export_AA
                    + AA_synthesis
                    - struct_synthesis
                    - storage_synthesis * self.r_AA_stor
                    + storage_catabolism / self.r_AA_stor
                    - AA_catabolism
                    - deficit_AA)
            if balance < 0.:
                deficit = - balance * (struct_mass) / self.time_step
                self.deficit_AA[vertex_index] = deficit if deficit > 1e-20 else 0.
                return 0.
            else:
                self.deficit_AA[vertex_index] = 0.
                return balance

        else:
            return 0

    @state
    def _storage_protein(self, storage_protein, struct_mass, storage_synthesis, storage_catabolism):
        if struct_mass > 0:
            return storage_protein + (self.time_step / struct_mass) * (
                    storage_synthesis
                    - storage_catabolism
            )
        else:
            return 0

    @state
    def _xylem_Nm(self, xylem_Nm, displaced_Nm_in, displaced_Nm_out, cumulated_radial_exchanges_Nm, struct_mass):
        if struct_mass > 0:
            # Vessel's nitrogen pool update
            # Xylem balance accounting for exports from all neighbors accessible by water flow
            return xylem_Nm + (displaced_Nm_in - displaced_Nm_out + cumulated_radial_exchanges_Nm) / struct_mass
        else:
            return 0

    @state
    def _xylem_AA(self, xylem_AA, displaced_AA_in, displaced_AA_out, cumulated_radial_exchanges_AA, struct_mass):
        if struct_mass > 0:
            return xylem_AA + (displaced_AA_in - displaced_AA_out + cumulated_radial_exchanges_AA) / struct_mass
        else:
            return 0

    # PLANT SCALE PROPERTIES UPDATE

    @totalstate
    def _total_phloem_AA(self, total_phloem_AA, diffusion_AA_phloem, AA_root_shoot_phloem):
        return total_phloem_AA[1] + (- self.time_step * sum(diffusion_AA_phloem.values()) + AA_root_shoot_phloem[1])
    
    @totalstate
    def _AA_root_shoot_phloem_record(self, AA_root_shoot_phloem):
        return AA_root_shoot_phloem[1]

    @totalstate
    def _total_cytokinins(self, total_cytokinins, cytokinin_synthesis, cytokinins_root_shoot_xylem):
        return total_cytokinins[1] + cytokinin_synthesis[1] * self.time_step - cytokinins_root_shoot_xylem[1]

    @totalstate
    def _C_Nm_average(self, Nm, struct_mass, total_living_struct_mass):
        return sum([x*y for x, y in zip(Nm.values(), struct_mass.values())]) / total_living_struct_mass[1]

    @totalstate
    def _C_AA_average(self, AA, struct_mass, total_living_struct_mass):
        return sum([x * y for x, y in zip(AA.values(), struct_mass.values())]) / total_living_struct_mass[1]

    @totalstate
    def _C_xylem_Nm_average(self, xylem_Nm, struct_mass, total_living_struct_mass):
        return sum([x*y for x, y in zip(xylem_Nm.values(), struct_mass.values())])  / total_living_struct_mass[1]

    @totalstate
    def _C_xylem_AA_average(self, xylem_AA, struct_mass, total_living_struct_mass):
        return sum([x*y for x, y in zip(xylem_AA.values(), struct_mass.values())])  / total_living_struct_mass[1]

    @totalstate
    def _total_AA_rhizodeposition(self, diffusion_AA_soil, import_AA):
        return self.time_step * (sum(diffusion_AA_soil.values()) - sum(import_AA.values()))

    @totalstate
    def _C_hexose_average(self, struct_mass, total_living_struct_mass, C_hexose_root=1e-4):
        return sum([x*y for x, y in zip(C_hexose_root.values(), struct_mass.values())]) / total_living_struct_mass[1]
