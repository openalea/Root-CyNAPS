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
from dataclasses import dataclass, field, fields
import inspect as ins
from functools import partial

from generic_fspm.component import Model, declare
from generic_fspm.component_factory import *



class RootNitrogenModel(Model):
    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

    # FROM SOIL MODEL
    soil_Nm: float = declare(default=0, unit="mol.g-1", unit_comment="of nitrates", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="input", by="model_soil", state_variable_type="", edit_by="user")
    soil_AA: float = declare(default=0, unit="mol.g-1", unit_comment="of amino acids", description="",
                            min_value="", max_value="", value_comment="", references="", variable_type="input", by="model_soil", 
                            state_variable_type="", edit_by="user")

    # FROM ANATOMY MODEL
    root_exchange_surface: float = declare(default=0, unit="m2", unit_comment="of cell membrane", description="",
                                           min_value="", max_value="", value_comment="", references="",  DOI="", 
                                           variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    stele_exchange_surface: float = declare(default=0, unit="m2", unit_comment="of cell membrane", description="", 
                                            min_value="", max_value="", value_comment="", references="",  DOI="", 
                                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    phloem_exchange_surface: float = declare(default=0, unit="m2", unit_comment="of vessel membrane", description="",
                                            min_value="", max_value="", value_comment="", references="",  DOI="", 
                                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    living_root_hairs_external_surface: float = declare(default=0, unit="m2", unit_comment="of root hair", description="", 
                                                        min_value="", max_value="", value_comment="", references="", DOI="",
                                                        variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    apoplasmic_stele: float = declare(default=0.5, unit="adim.", unit_comment="", description="", 
                                      min_value="", max_value="", value_comment="", references="", DOI="",
                                      variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")

    # FROM CARBON BALANCE MODEL
    C_hexose_root: float =     declare(default=0, unit="mol.g-1", unit_comment="of hexose", description="",
                                       min_value="", max_value="", value_comment="", references="", DOI="",
                                       variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    C_hexose_reserve: float =  declare(default=0, unit="mol.g-1", unit_comment="of hexose", description="", 
                                       min_value="", max_value="", value_comment="", references="", DOI="",
                                       variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")

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
    struct_mass_produced: float =       declare(default=0, unit="g", unit_comment="of dry weight", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    thermal_time_since_emergence: float = declare(default=0, unit="°C", unit_comment="", description="", 
                                                  min_value="", max_value="", value_comment="", references="", DOI="",
                                                  variable_type="input", by="model_growth", state_variable_type="", edit_by="user")

    # FROM SHOOT MODEL
    AA_root_shoot_phloem: float =       declare(default=0, unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    cytokinins_root_shoot_xylem: float = declare(default=0, unit="mol.s-1", unit_comment="of cytokinins", description="", 
                                                 min_value="", max_value="", value_comment="", references="", DOI="",
                                                 variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")

    # --- INITIALIZE MODEL STATE VARIABLES ---

    # LOCAL VARIABLES

    # Pools initial size
    Nm: float =                 declare(default=1e-4, unit="mol.g-1", unit_comment="of nitrates", description="", 
                                        min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="intensive", edit_by="user")
    AA: float =                 declare(default=9e-4, unit="mol.g-1", unit_comment="of amino acids", description="", 
                                        min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="intensive", edit_by="user")
    struct_protein: float =     declare(default=0., unit="mol.g-1", unit_comment="of structural proteins", description="", 
                                        min_value="", max_value="", value_comment="0 value for wheat", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="intensive", edit_by="user")
    storage_protein: float =    declare(default=0., unit="mol.g-1", unit_comment="of storage proteins", description="", 
                                        min_value="", max_value="", value_comment="0 value for wheat", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="intensive", edit_by="user")
    xylem_Nm: float =           declare(default=1e-4, unit="mol.g-1", unit_comment="of structural nitrates", description="", 
                                        min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="intensive", edit_by="user")
    xylem_AA: float =           declare(default=1e-4, unit="mol.g-1", unit_comment="of amino acids", description="", 
                                        min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="state_variable", by="model_nitrogen", state_variable_type="intensive", edit_by="user")
    # Transport processes
    import_Nm: float =                      declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="", 
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    import_AA: float =                      declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    export_Nm: float =                      declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    export_AA: float =                      declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    diffusion_Nm_soil: float =              declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    diffusion_Nm_xylem: float =             declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="",
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="", 
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    diffusion_Nm_soil_xylem: float =        declare(default=0., unit="mol.s-1", unit_comment="of nitrates", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    diffusion_AA_soil: float =              declare(default=0., unit="mol.s-1", unit_comment="of amino acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    diffusion_AA_phloem: float =            declare(default=0., unit="mol.s-1", unit_comment="of amino acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    diffusion_AA_soil_xylem: float =        declare(default=0., unit="mol.s-1", unit_comment="of amino acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    # Metabolic processes
    AA_synthesis: float =                   declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    struct_synthesis: float =               declare(default=0., unit="mol.s-1", unit_comment="of functional structure", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    storage_synthesis: float =              declare(default=0., unit="mol.s-1", unit_comment="of storage", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    AA_catabolism: float =                  declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    storage_catabolism: float =             declare(default=0., unit="mol.s-1", unit_comment="of storage", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    xylem_struct_mass: float =              declare(default=1e-6, unit="g", unit_comment="of structural mass", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    phloem_struct_mass: float =              declare(default=1e-6, unit="g", unit_comment="of structural mass", description="", 
                                                     min_value="", max_value="", value_comment="", references="", DOI="",
                                                     variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    displaced_Nm_in: float =                declare(default=0., unit="mol.time_step-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    displaced_Nm_out: float =               declare(default=0., unit="mol.time_step-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    displaced_AA_in: float =                declare(default=0., unit="mol.time_step-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    displaced_AA_out: float =               declare(default=0., unit="mol.time_step-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    cumulated_radial_exchanges_Nm: float =  declare(default=0., unit="mol.time_step-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    cumulated_radial_exchanges_AA: float =  declare(default=0., unit="mol.time_step-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_nitrogen", state_variable_type="extensive", edit_by="user")

    # SUMMED STATE VARIABLES

    total_Nm: float =                   declare(default=0., unit="mol.g-1", unit_comment="of nitrates", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    total_AA: float =                   declare(default=0., unit="mol.g-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    total_hexose: float =               declare(default=0., unit="mol.g-1", unit_comment="of hexose", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    total_cytokinins: float =           declare(default=100, unit="UA.s-1", unit_comment="of cytokinins", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    total_struct_mass: float =          declare(default=0., unit="g", unit_comment="of dry weight", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    total_xylem_Nm: float =             declare(default=0., unit="mol.g-1", unit_comment="of nitrates", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    total_xylem_AA: float =             declare(default=0., unit="mol.g-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    total_phloem_AA: float =            declare(default=0, unit="mol.g-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    Nm_root_shoot_xylem: float =        declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    AA_root_shoot_xylem: float =        declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_nitrogen", state_variable_type="", edit_by="user")
    total_AA_rhizodeposition: float =   declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
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
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    vmax_Nm_xylem: float =              declare(default=1e-6, unit="mol.s-1.m-2", unit_comment="of nitrates", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_Nm_root_LATS: float =            declare(default=1e-1, unit="mol.m-3", unit_comment="of nitrates", description="", 
                                                min_value="", max_value="", value_comment="Changed to increase diminution", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_Nm_root_HATS: float =            declare(default=1e-6, unit="mol.m-3", unit_comment="of nitrates", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    begin_N_regulation: float =         declare(default=1e1, unit="mol.g-1", unit_comment="of nitrates", description="", 
                                                min_value="", max_value="", value_comment="changed so that import_Nm variation may occur in Nm variation range", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    span_N_regulation: float =          declare(default=2e-4, unit="mol.g-1", unit_comment="of nitrates", description="", 
                                                min_value="", max_value="", value_comment="range corresponding to observed variation range within segment", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_Nm_xylem: float =                declare(default=8e-5, unit="mol.g-1", unit_comment="of nitrates", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    vmax_AA_root: float =               declare(default=1e-7, unit="mol.s-1.m-2", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_AA_root: float =                 declare(default=1e-1, unit="mol.m-3", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    vmax_AA_xylem: float =              declare(default=1e-7, unit="mol.s-1.m-2", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_AA_xylem: float =                declare(default=8e-5, unit="mol.g-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    diffusion_soil: float =             declare(default=1e-11, unit="g.s-1.m-2", unit_comment="of solute", description="", 
                                                min_value="", max_value="", value_comment="while there is no soil model balance", references="", DOI="", 
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    diffusion_xylem: float =            declare(default=0., unit="g.s-1.m-2", unit_comment="of solute", description="", 
                                                min_value="", max_value="", value_comment="from 1e-6, It was noticed it only contributed to xylem loading", references="", DOI="", 
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    diffusion_phloem: float =           declare(default=1e-5, unit="g.s-1.m-2", unit_comment="of solute", description="", 
                                                min_value="", max_value="", value_comment="more realistic ranges than?", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")  # Artif *1e-1 g.m-2.s-1 more realistic ranges
    diffusion_apoplasm: float =         declare(default=2.5e-10, unit="g.s-1.m-2", unit_comment="of solute", description="", 
                                                min_value="", max_value="", value_comment="while there is no soil model balance", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")

    # metabolism-related parameters
    transport_C_regulation: float =     declare(default=7e-3, unit="mol.g-1", unit_comment="of hexose", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")

    # N METABOLISM PROCESSES
    # TODO : introduce nitrogen fixation
    # kinetic parameters
    smax_AA: float =                    declare(default=1e-5, unit="mol.s-1.g-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_Nm_AA: float =                   declare(default=3e-6, unit="mol.g-1", unit_comment="of nitrates", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_C_AA: float =                    declare(default=350e-6, unit="mol.g-1", unit_comment="of hexose", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    smax_struct: float =                declare(default=1e-9, unit="mol.s-1.g-1", unit_comment="of structure", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_AA_struct: float =               declare(default=250e-6, unit="mol.g-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    smax_stor: float =                  declare(default=0., unit="mol.s-1.g-1", unit_comment="of storage", description="", 
                                                min_value="", max_value="", value_comment="0 for wheat", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_AA_stor: float =                 declare(default=250e-6, unit="mol.g-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    cmax_stor: float =                  declare(default=1e-9, unit="mol.s-1.g-1", unit_comment="of storage", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_stor_catab: float =              declare(default=250e-6, unit="mol.g-1", unit_comment="of storage", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    cmax_AA: float =                    declare(default=0., unit="mol.s-1.g-1", unit_comment="of amino acids", description="", 
                                                min_value="", max_value="", value_comment="5e-9 for now not relevant as it doesn't contribute to C_hexose_root balance", references="", DOI="",
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
    Km_C_cytok: float =                 declare(default=1.2e-3, unit="mol.g-1", unit_comment="of hexose", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    Km_N_cytok: float =                 declare(default=1.2e-3, unit="mol.g-1", unit_comment="of nitrates", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")

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
    xylem_cross_area_ratio: float =     declare(default=0.84 * (0.36 ** 2), unit="adim", unit_comment="surface ratio", description="xylem cross-section area ratio * stele radius ratio^2", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")
    phloem_cross_area_ratio: float =    declare(default=0.15 * (0.36 ** 2), unit="adim", unit_comment="surface ratio", description="phloem cross-section area ratio * stele radius ratio^2", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="parameter", by="model_nitrogen", state_variable_type="", edit_by="user")

    def __init__(self, g, time_step, **scenario) -> None:

        """
        Description
        Initialization of nitrogen-related variables

        Parameters
        :param g: MTG
        """
        self.g = g
        self.props = self.g.properties()
        self.executor.add_data(self.props)
        self.time_step = time_step
        self.vertices = self.g.vertices(scale=self.g.max_scale())

        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)
        self.link_self_to_mtg()

    @postgrowth
    def post_growth_updating(self):
        """
        Description :
            Extend property dictionnary uppon new element partionning and updates concentrations uppon structural_mass change
        """
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        for vid in self.vertices:
            if vid not in list(self.Nm.keys()):
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
            else:
                for prop in self.state_variables:
                    # if intensive, concentrations have to be updated based on new structural mass
                    if self.__dataclass_fields__[prop].metadata["state_variable_type"] == "intensive":
                        getattr(self, prop).update({vid: getattr(self, prop)[vid] * (
                                self.initial_struct_mass[vid] / self.struct_mass[vid])})

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
    def _import_Nm(self, Nm, soil_Nm, root_exchange_surface, C_hexose_root, living_root_hairs_external_surface):
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
        return ((soil_Nm * self.vmax_Nm_root / (soil_Nm + Km_Nm_root))
                * (root_exchange_surface + living_root_hairs_external_surface)
                * (C_hexose_root / (C_hexose_root + self.transport_C_regulation)))

    @rate
    def _diffusion_Nm_soil(self, Nm, soil_Nm, root_exchange_surface, living_root_hairs_external_surface):
        # Passive radial diffusion between soil and cortex.
        # It happens only through root segment external surface.
        # We summarize apoplasm-soil and cortex-soil diffusion in 1 flow.
        return (self.diffusion_soil * (Nm * 10e5 - soil_Nm) * (
                root_exchange_surface + living_root_hairs_external_surface))

    @rate
    def _export_Nm(self, Nm, stele_exchange_surface, C_hexose_root):
        # We define active export to xylem from root segment
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        return ((Nm * self.vmax_Nm_xylem) / (Nm + self.Km_Nm_xylem)) * stele_exchange_surface * (
                C_hexose_root / (C_hexose_root + self.transport_C_regulation))

    @rate
    def _diffusion_Nm_xylem(self, xylem_Nm, Nm, stele_exchange_surface):
        # Passive radial diffusion between xylem and cortex through plasmalema
        return self.diffusion_xylem * (xylem_Nm - Nm) * stele_exchange_surface

    @rate
    def _diffusion_Nm_soil_xylem(self, soil_Nm, xylem_Nm, radius, length, xylem_differentiation_factor, endodermis_conductance_factor):
        # Direct diffusion between soil and xylem when 1) xylem is apoplastic and 2) endoderm is not differentiated
        # Here, surface is not really representative of a structure as everything is apoplasmic
        return self.diffusion_apoplasm * (
                soil_Nm - xylem_Nm * 10e5) * 2 * np.pi * radius * length * xylem_differentiation_factor * endodermis_conductance_factor

    # AMINO ACID TRANSPORT
    @rate
    def _import_AA(self, soil_AA, root_exchange_surface, living_root_hairs_external_surface, C_hexose_root):
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        return ((soil_AA * self.vmax_AA_root / (soil_AA + self.Km_AA_root))
                * (root_exchange_surface + living_root_hairs_external_surface)
                * (C_hexose_root / (C_hexose_root + self.transport_C_regulation)))

    @rate
    def _diffusion_AA_soil(self, AA, soil_AA, root_exchange_surface, living_root_hairs_external_surface):
        # We define amino acid passive diffusion to soil
        return (self.diffusion_soil * (AA * 10e5 - soil_AA)
                * (root_exchange_surface + living_root_hairs_external_surface))

    @rate
    def _export_AA(self, AA, stele_exchange_surface, C_hexose_root):
        # We define active export to xylem from root segment
        # Km is defined as a constant here
        # (Michaelis-Menten kinetic, surface dependency, active transport C requirements)
        return ((AA * self.vmax_AA_xylem / (AA + self.Km_AA_xylem))
                * stele_exchange_surface * (C_hexose_root / (
                        C_hexose_root + self.transport_C_regulation)))

    @rate
    def _diffusion_AA_soil_xylem(self, soil_AA, xylem_AA, radius, length, xylem_differentiation_factor, endodermis_conductance_factor):
        # Direct diffusion between soil and xylem when 1) xylem is apoplastic and 2) endoderm is not differentiated
        return (self.diffusion_apoplasm * (soil_AA - xylem_AA * 10e5)
                * 2 * np.pi * radius * length * xylem_differentiation_factor * endodermis_conductance_factor)

    @rate
    def _diffusion_AA_phloem(self, AA, phloem_exchange_surface):
        # Passive radial diffusion between phloem and cortex through plasmodesmata
        # TODO : Change diffusive flow to enable realistic ranges, now, unloading is limited by a ping pong bug related to diffusion
        # TODO : resolve exception when mapping has to deal with plant scale properties AND local ones
        return (self.diffusion_phloem * (self.total_phloem_AA[1] - AA)
                * phloem_exchange_surface)

    @axial
    @rate
    # AXIAL TRANSPORT PROCESSES
    def _axial_transport_N(self, index):
        """
                Description
                ___________

        """
        # TODO : probably collar children have to be reintroduced for good neighbor management. (from model_water)
        #  But if null water content proprely passes information between collar and it's children,
        #  it may be already working well

        # AXIAL TRANSPORT
        v = index

        # If this is only an out flow to up parents
        if self.axial_export_water_up[v] > 0:
            # Turnover defines a dilution factor of radial transport processes over the axially transported
            # water column
            turnover = self.axial_export_water_up[v] / self.xylem_water[v]
            if turnover <= 1:
                # Transport only affects considered segment
                self.cumulated_radial_exchanges_Nm[v] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * self.time_step
                self.cumulated_radial_exchanges_AA[v] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * self.time_step
                # Exported matter corresponds to the exported water proportion
                self.displaced_Nm_out[v] = turnover * self.xylem_Nm[v] * self.xylem_struct_mass[v]
                self.displaced_AA_out[v] = turnover * self.xylem_AA[v] * self.xylem_struct_mass[v]
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
                self.displaced_Nm_out[v] = self.xylem_Nm[v] * self.xylem_struct_mass[v]
                self.displaced_AA_out[v] = self.xylem_AA[v] * self.xylem_struct_mass[v]
                # Transport affects a chain of parents
                water_exchange_time = self.time_step / turnover
                # Loading of the current vertex into the current vertex's xylem
                self.cumulated_radial_exchanges_Nm[v] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time
                self.cumulated_radial_exchanges_AA[v] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time

                exported_water = self.axial_export_water_up[v]
                child = v
                # Loading of the current vertex into the vertices who have received water from it
                while exported_water > 0:
                    # We remove the amount of water which has already received loading in previous loop
                    exported_water -= self.xylem_water[child]
                    up_parent = self.g.parent(child)
                    # If we reached collar, this amount is being exported
                    if up_parent == None:
                        self.Nm_root_shoot_xylem[1] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * exported_water / self.xylem_water[v]
                        self.AA_root_shoot_xylem[1] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * exported_water / self.xylem_water[v]
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
                            self.cumulated_radial_exchanges_Nm[up_parent] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * self.xylem_water[up_parent] / self.xylem_water[v]
                            self.cumulated_radial_exchanges_AA[up_parent] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * self.xylem_water[up_parent] / self.xylem_water[v]
                        # If it's only partial, we account only for the exceeding amount
                        else:
                            self.cumulated_radial_exchanges_Nm[up_parent] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * exported_water / self.xylem_water[v]
                            self.cumulated_radial_exchanges_AA[up_parent] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * exported_water / self.xylem_water[v]
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
        if self.axial_import_water_down[v] < 0:
            # Turnover defines a dilution factor of radial transport processes over the axially transported
            # water column
            turnover = - self.axial_import_water_down[v] / self.xylem_water[v]
            if turnover <= 1:
                # Transport only affects considered segment
                self.cumulated_radial_exchanges_Nm[v] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * self.time_step
                self.cumulated_radial_exchanges_AA[v] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * self.time_step
                # Exported matter corresponds to the exported water proportion
                self.displaced_Nm_out[v] = turnover * self.xylem_Nm[v] * self.xylem_struct_mass[v]
                self.displaced_AA_out[v] = turnover * self.xylem_AA[v] * self.xylem_struct_mass[v]
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
                self.displaced_Nm_out[v] = self.xylem_Nm[v] * self.xylem_struct_mass[v]
                self.displaced_AA_out[v] = self.xylem_AA[v] * self.xylem_struct_mass[v]
                # Transport affects a chain of children
                water_exchange_time = self.time_step / turnover
                # Loading of the current vertex into the current vertex's xylem
                self.cumulated_radial_exchanges_Nm[v] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time
                self.cumulated_radial_exchanges_AA[v] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time

                parent = [v]
                # We initialize a list tracking water repartition among down axes
                axis_proportion = [1.0]
                # We remove the amount of water which has already been received
                exported_water = [-self.axial_import_water_down[v] - self.xylem_water[v]]
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
                                self.cumulated_radial_exchanges_Nm[parent[p]] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * exported_water[p] / self.xylem_water[v]
                                self.cumulated_radial_exchanges_AA[parent[p]] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * exported_water[p] / self.xylem_water[v]
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
                                    self.cumulated_radial_exchanges_Nm[down_children[0]] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * self.xylem_water[down_children[0]] / self.xylem_water[v]
                                    self.cumulated_radial_exchanges_AA[down_children[0]] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * self.xylem_water[down_children[0]] / self.xylem_water[v]
                                    children_exported_water += [exported_water[p] - self.xylem_water[down_children[0]]]
                                else:
                                    self.cumulated_radial_exchanges_Nm[down_children[0]] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * exported_water[p] / self.xylem_water[v]
                                    self.cumulated_radial_exchanges_AA[down_children[0]] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * exported_water[p] / self.xylem_water[v]
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
                                        self.cumulated_radial_exchanges_Nm[down_children[ch]] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * self.xylem_water[down_children[ch]] / self.xylem_water[v]
                                        self.cumulated_radial_exchanges_AA[down_children[ch]] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * self.xylem_water[down_children[ch]] / self.xylem_water[v]
                                        children_down_flow[ch] -= self.xylem_water[down_children[ch]]
                                    else:
                                        self.cumulated_radial_exchanges_Nm[down_children[ch]] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * water_exchange_time * children_down_flow[ch] / self.xylem_water[v]
                                        self.cumulated_radial_exchanges_AA[down_children[ch]] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * water_exchange_time * children_down_flow[ch] / self.xylem_water[v]
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
        if self.axial_import_water_down[v] >= 0 >= self.axial_export_water_up[v]:
            # There is no exported matter, thus no receiver
            self.displaced_Nm_out[v] = 0
            self.displaced_AA_out[v] = 0
            # No matter what the transported water amount is, all radial transport effects remain on the current vertex.
            self.cumulated_radial_exchanges_Nm[v] += (self.export_Nm[v] + self.diffusion_Nm_soil_xylem[v] - self.diffusion_Nm_xylem[v]) * self.time_step
            self.cumulated_radial_exchanges_AA[v] += (self.export_AA[v] + self.diffusion_AA_soil_xylem[v]) * self.time_step

    # METABOLIC PROCESSES
    def _AA_synthesis(self, C_hexose_root, struct_mass, Nm):
        # amino acid synthesis
        if C_hexose_root > 0 and Nm > 0:
            return struct_mass * self.smax_AA / (
                    ((1 + self.Km_Nm_AA) / Nm) + ((1 + self.Km_C_AA) / C_hexose_root))
        else:
            return 0

    # def _struct_synthesis(self, v, smax_struct, Km_AA_struct):
    #     # Organic structure synthesis (REPLACED BY RHIZODEP struct_mass_produced)
    #     struct_synthesis = struct_mass * (smax_struct * AA / (Km_AA_struct + AA))
        
    @rate
    def _storage_synthesis(self, struct_mass, AA):
        # Organic storage synthesis (Michaelis-Menten kinetic)
        return struct_mass * (self.smax_stor * AA / (self.Km_AA_stor + AA))

    @rate
    def _storage_catabolism(self, struct_mass, C_hexose_root, C_hexose_reserve):
        # Organic storage catabolism through proteinase
        Km_stor_root = self.Km_stor_catab * np.exp(self.storage_C_regulation * C_hexose_root)
        return struct_mass * self.cmax_stor * C_hexose_reserve / (Km_stor_root + C_hexose_reserve)

    @rate
    def _AA_catabolism(self, C_hexose_root, struct_mass, AA):
        # AA catabolism through GDH
        Km_stor_root = self.Km_AA_catab * np.exp(self.storage_C_regulation * C_hexose_root)
        return struct_mass * self.cmax_AA * AA / (Km_stor_root + AA)

    @rate
    def _cytokinin_synthesis(self, total_struct_mass, total_hexose, total_Nm):
        return total_struct_mass * self.smax_cytok * (
                total_hexose / (total_hexose + self.Km_C_cytok)) * (
                total_Nm / (total_Nm + self.Km_N_cytok))

    @state
    # UPDATE NITROGEN POOLS
    def _Nm(self, Nm, struct_mass, import_Nm, diffusion_Nm_soil, diffusion_Nm_xylem, export_Nm, AA_synthesis, AA_catabolism):
        if struct_mass > 0:
            return Nm + (self.time_step / struct_mass) * (
                    import_Nm
                    - diffusion_Nm_soil
                    + diffusion_Nm_xylem
                    - export_Nm
                    - AA_synthesis * self.r_Nm_AA
                    + AA_catabolism / self.r_Nm_AA)
        else:
            return 0

    @state
    def _AA(self, AA, struct_mass, diffusion_AA_phloem, import_AA, diffusion_AA_soil, export_AA, AA_synthesis,
                  struct_synthesis, storage_synthesis, storage_catabolism, AA_catabolism, struct_mass_produced):
        if struct_mass > 0:
            return AA + (self.time_step / struct_mass) * (
                    diffusion_AA_phloem
                    + import_AA
                    - diffusion_AA_soil
                    - export_AA
                    + AA_synthesis
                    - struct_synthesis * self.r_AA_struct
                    - storage_synthesis * self.r_AA_stor
                    + storage_catabolism / self.r_AA_stor
                    - AA_catabolism
            ) - struct_mass_produced * 0.2 / 146
        # glutamine 5 C -> 60g.mol-1 2N -> 28 g.mol-1 : C:N = 2.1
        # Sachant C:N struct environ de 10 = (Chex + CAA)/NAA Chex = 10*28 - 60 = 220 g Chex.
        # Sachang qu'un hexose contient 12*6=72 gC.mol-1 hex, c'est donc environ 3 hexoses pour 1 AA qui seraient consommés.
        # La proportion d'AA consommée par g de struct mass est donc de 1*146/(3*180 + 1*146) = 0.2 (180 g.mol-1 pour le glucose)

        else:
            return 0

    # def _struct_protein(self, v):
    #     struct_protein += (sub_time_step / struct_mass) * (
    #         struct_synthesis)

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

    @state
    # UPDATE STRUCTURAL VALUES TODO : do not keep in this module
    def _xylem_struct_mass(self, struct_mass):
        return struct_mass * self.xylem_cross_area_ratio

    @state
    def _phloem_struct_mass(self, struct_mass):
        return struct_mass * self.phloem_cross_area_ratio

    @state
    # CARBON UPDATE
    def _C_hexose_root(self, C_hexose_root):
        # Minimum to avoid issues with zero values
        if C_hexose_root <= 0:
            return 1e-1
        else:
            return C_hexose_root

    @totalstate
    # PLANT SCALE PROPERTIES UPDATE
    def _total_phloem_AA(self, total_phloem_AA, diffusion_AA_phloem, AA_root_shoot_phloem, total_struct_mass):
        return total_phloem_AA[1] + (- self.time_step * sum(diffusion_AA_phloem.values()) + AA_root_shoot_phloem[1]) / (
                total_struct_mass[1] * self.phloem_cross_area_ratio)

    @totalstate
    def _total_cytokinins(self, total_cytokinins, cytokinin_synthesis, cytokinins_root_shoot_xylem,
                                   total_struct_mass):
        return total_cytokinins[1] + (cytokinin_synthesis[1] * self.time_step -
                                     cytokinins_root_shoot_xylem[1]) / total_struct_mass[1]

    # UPDATE CUMULATIVE VALUES
    # TODO : Retrieve the total struct mass from Rhizodep, otherwise, computation order is messed up here.

    @totalstate
    def _total_struct_mass(self, struct_mass):
        # WARNING, do not parallelize otherwise other pool updates will be based on previous time-step
        return sum(struct_mass.values())

    @totalstate
    def _total_Nm(self, Nm, struct_mass, total_struct_mass):
        return sum([x*y for x, y in zip(Nm.values(), struct_mass.values())]) / total_struct_mass[1]

    @totalstate
    def _total_AA(self, AA, struct_mass, total_struct_mass):
        return sum([x * y for x, y in zip(AA.values(), struct_mass.values())]) / total_struct_mass[1]

    @totalstate
    def _total_xylem_Nm(self, xylem_Nm, xylem_struct_mass, total_struct_mass):
        return sum([x*y for x, y in zip(xylem_Nm.values(), xylem_struct_mass.values())]) / total_struct_mass[1]

    @totalstate
    def _total_xylem_AA(self, xylem_AA, xylem_struct_mass, total_struct_mass):
        return sum([x*y for x, y in zip(xylem_AA.values(), xylem_struct_mass.values())]) / total_struct_mass[1]

    @totalstate
    def _total_AA_rhizodeposition(self, diffusion_AA_soil, import_AA):
        return self.time_step * (sum(diffusion_AA_soil.values()) - sum(import_AA.values()))

    @totalstate
    def _total_hexose(self, C_hexose_root, struct_mass, total_struct_mass):
        return sum([x*y for x, y in zip(C_hexose_root.values(), struct_mass.values())]) / total_struct_mass[1]
