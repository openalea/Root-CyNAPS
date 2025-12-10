# Public packages
import numpy as np
from dataclasses import dataclass
import cmf
import time
from multiprocessing.shared_memory import SharedMemory
import numpy as np

# Utility packages
from openalea.metafspm.component_factory import *
from openalea.metafspm.component import Model, declare


@dataclass
class SoilModel(Model):
    """
    Empty doc
    """

    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---
    
    # FROM ANATOMY MODEL
    root_exchange_surface: float = declare(default=0., unit="m2", unit_comment="", description="Exchange surface between soil and symplasmic parenchyma.", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")

    # FROM GROWTH MODEL
    length: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example root segment length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    initial_length: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example root segment length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")

    # FROM NITROGEN MODEL
    mineralN_uptake: float = declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="", 
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    amino_acids_uptake: float = declare(default=0., unit="mol.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    mineralN_diffusion_from_roots: float =  declare(default=0., unit="mol.s-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    amino_acids_diffusion_from_roots: float =  declare(default=0., unit="mol.s-1", unit_comment="of amino acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    mineralN_diffusion_from_xylem: float =  declare(default=0., unit="mol.s-1", unit_comment="of nitrates", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    amino_acids_diffusion_from_xylem: float =  declare(default=0., unit="mol.s-1", unit_comment="of amino_acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
   


    # FROM WATER MODEL
    water_uptake: float =  declare(default=0., unit="m3.s-1", unit_comment="of water", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_water", state_variable_type="extensive", edit_by="user")
    
    # FROM METEO
    water_irrigation: float =  declare(default=10/(24*3600), unit="g.s-1", unit_comment="of water", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="plant_scale_state", by="meteo", state_variable_type="extensive", edit_by="user")
    water_evaporation: float =  declare(default=5/(24*3600), unit="g.s-1", unit_comment="of water", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="plant_scale_state", by="meteo", state_variable_type="extensive", edit_by="user")
    water_drainage: float =  declare(default=5/(24*3600), unit="g.s-1", unit_comment="of water", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="plant_scale_state", by="meteo", state_variable_type="extensive", edit_by="user")
    voxel_mineral_N_fertilization: float =  declare(default=0., unit="g.s-1", unit_comment="of nitrogen", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="meteo", state_variable_type="extensive", edit_by="user")
    mineral_N_fertilization_rate: float =  declare(default=0., unit="g.s-1", unit_comment="of nitrogen", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="meteo", state_variable_type="extensive", edit_by="user")

    # --- @note STATE VARIABLES INITIALIZATION ---
    # Temperature
    soil_temperature: float = declare(default=7.8, unit="°C", unit_comment="", description="soil temperature in contact with roots",
                                                 value_comment="Derived from Swinnen et al. 1994 C inputs, estimated from a labelling experiment starting 3rd of March, with average temperature at 7.8 °C", references="Swinnen et al. 1994", DOI="",
                                                 min_value="", max_value="", variable_type="state_variable", by="model_temperature", state_variable_type="intensive", edit_by="user")

    # C related
    POC: float = declare(default=2.e-3, unit="adim", unit_comment="gC per g of dry soil", description="Particulate Organic Carbon massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    MAOC: float = declare(default=8.e-3, unit="adim", unit_comment="gC per g of dry soil", description="Mineral Associated Organic Carbon in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    DOC: float = declare(default=2e-7, unit="adim", unit_comment="gC per g of dry soil", description="Dissolved Organic Carbon massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    microbial_C: float = declare(default=0.2e-3, unit="adim", unit_comment="gC per g of dry soil", description="microbial Carbon massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    CO2: float = declare(default=0, unit="adim", unit_comment="gC per g of dry soil", description="Carbon dioxyde massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    C_hexose_soil: float = declare(default=2.4e-3, unit="mol.m-3", unit_comment="of hexose", description="Hexose concentration in soil", 
                                        value_comment="", references="Fischer et al 2007, water leaching estimation", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    content_hexose_soil: float = declare(default=2.4e-3, unit="mol.g-1", unit_comment="of hexose", description="Hexose concentration in soil", 
                                        value_comment="", references="Fischer et al 2007, water leaching estimation", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Cs_mucilage_soil: float = declare(default=0., unit="mol.m-3", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Cs_cells_soil: float = declare(default=0., unit="mol.m-3", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    # N related
    PON: float = declare(default=0.1e-3, unit="adim", unit_comment="gN per g of dry soil", description="Particulate Organic Nitrogen massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    MAON: float = declare(default=0.8e-3, unit="adim", unit_comment="gN per g of dry soil", description="Mineral-Associated Organic Nitrogen massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    DON: float = declare(default=2e-8, unit="adim", unit_comment="gN per g of dry soil", description="Dissolved Organic Nitrogen massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    microbial_N: float = declare(default=0.03e-3, unit="adim", unit_comment="gN per g of dry soil", description="microbial N massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Nm_fungus: float = declare(default=0., unit="adim", unit_comment="gN per g of dry soil", description="mycorrhiza N massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    dissolved_mineral_N: float = declare(default=20e-6, unit="adim", unit_comment="gN per g of dry soil", description="dissolved mineral N massic concentration in soil",
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")

    C_mineralN_soil: float = declare(default=2.2, unit="mol.m-3", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    C_amino_acids_soil: float = declare(default=8.2e-3, unit="mol.m-3", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in soil", 
                                        value_comment="", references="Fischer et al 2007, water leaching estimation", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    
    # All solutes
    Cv_solutes_soil: float = declare(default=32.2 / 10, unit="mol.m-3", unit_comment="mol of  all dissolved mollecules in the soil solution", description="All dissolved mollecules concentration", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")

    # Water related
    water_potential_soil: float = declare(default=-0.1e6, unit="Pa", unit_comment="", description="Mean soil water potential", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    soil_moisture: float = declare(default=0.3, unit="adim", unit_comment="g.g-1", description="Volumetric proportion of water per volume of soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    water_volume: float = declare(default=0.25e-6, unit="m3", unit_comment="", description="Volume of the water in the soil element in contact with a the root segment", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    
    
    # Structure related
    voxel_volume: float = declare(default=1e-6, unit="m3", unit_comment="", description="Volume of the soil element in contact with a the root segment",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    bulk_density: float = declare(default=1.42, unit="g.mL", unit_comment="", description="Volumic density of the dry soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    dry_soil_mass: float = declare(default=1.42, unit="g", unit_comment="", description="dry weight of the considered voxel element", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    
    # --- @note RATES INITIALIZATION ---
    # In-voxel rates
    microbial_activity: float = declare(default=0., unit="adim", unit_comment="", description="microbial degradation activity indicator depending on microbial activity locally", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    degradation_POC: float = declare(default=0., unit=".s-1", unit_comment="gC per g of soil per second", description="degradation rate of POC", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    degradation_MAOC: float = declare(default=0., unit=".s-1", unit_comment="gC per g of soil per second", description="degradation rate of MAOC", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    degradation_DOC: float = declare(default=0., unit=".s-1", unit_comment="gC per g of soil per second", description="degradation rate of DOC", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    degradation_microbial_OC: float = declare(default=0., unit=".s-1", unit_comment="gC per g of soil per second", description="degradation rate of microbial OC", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    mineral_N_net_mineralization: float = declare(default=0., unit=".s-1", unit_comment="gN per g of soil per second", description="mineral N uptake by micro organisms", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    
    # Transport rates
    soil_water_flux: float = declare(default=0., unit="m.s-1", unit_comment="m3.m-2.s-1", description="volumetric water flux per surface area from Richards equations", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    mineral_N_transport: float = declare(default=0., unit="mol.s-1", unit_comment="", description="mineral N advection-dispersion flux derived from Richards water transport", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    amino_acid_transport: float = declare(default=0., unit="mol.s-1", unit_comment="", description="mineral N advection-dispersion flux derived from Richards water transport", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")

    
    # Degradation processes
    hexose_degradation: float = declare(default=0., unit="mol.s-1", unit_comment="", description="Rate of hexose consumption  at the soil-root interface", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    mucilage_degradation: float = declare(default=0., unit="mol.s-1", unit_comment="", description="Rate of mucilage degradation outside the root", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    cells_degradation: float = declare(default=0., unit="mol.s-1", unit_comment="", description="Rate of root cells degradation outside the root", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")

    # --- @note PARAMETERS ---

    # C related
    k_POC: float = declare(default=0.5 / (3600 * 24 * 365), unit=".s-1", unit_comment="", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    k_MAOC: float = declare(default=0.01 / (3600 * 24 * 365), unit=".s-1", unit_comment="", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    k_DOC: float = declare(default=20. / (3600 * 24 * 365), unit=".s-1", unit_comment="", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    k_MbOC: float = declare(default=10. / (3600 * 24 * 365), unit=".s-1", unit_comment="", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    
    microbial_C_min: float = declare(default=0.1, unit="adim", unit_comment="gC per g of dry soil", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    microbial_C_max: float = declare(default=0.5, unit="adim", unit_comment="gC per g of dry soil", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    microbial_proportion_of_MAOM: float = declare(default=0.1, unit="adim", unit_comment="gC POM per gC of microbial biomass", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    
    # N related
    CN_ratio_POM: float = declare(default=20, unit="adim", unit_comment="gC per gN", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    CN_ratio_MAOM: float = declare(default=10, unit="adim", unit_comment="gC per gN", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    CN_ratio_microbial_biomass: float = declare(default=11, unit="adim", unit_comment="gC per gN", description="", 
                                        value_comment="", references="Perveen et al. 2014", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    CN_ratio_root_cells: float = declare(default=8, unit="adim", unit_comment="gC per gN", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    
    CUE_POC: float = declare(default=0.2, unit="adim", unit_comment="gC per gC", description="Carbon Use efficiency of microorganism degradation for POC", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    CUE_MAOC: float = declare(default=0.4, unit="adim", unit_comment="gC per gC", description="Carbon Use efficiency of microorganism degradation for MAOC", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    CUE_DOC: float = declare(default=0.4, unit="adim", unit_comment="gC per gC", description="Carbon Use efficiency of microorganism degradation for DOC", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    CUE_MbOC: float = declare(default=0.3, unit="adim", unit_comment="gC per gC", description="Carbon Use efficiency of microorganism degradation for MBC", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    
    # max_N_uptake_per_microbial_C: float = declare(default=1e-7, unit=".s-1", unit_comment="gN per second per gC of microbial C", description="", 
    #                                     value_comment="", references="", DOI="",
    #                                    min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    # Km_microbial_N_uptake: float = declare(default=0.01 / 1e3, unit="adim", unit_comment="gN per g of dry soil", description="", 
    #                                     value_comment="", references="", DOI="",
    #                                    min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    
    ratio_C_per_amino_acid: float = declare(default=6, unit="adim", unit_comment="number of carbon per molecule of amino acid", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    CN_ratio_amino_acids: float = declare(default=6/1.4, unit="adim", unit_comment="", description="CN ratio of amino acids (6 C and 1.4 N on average)", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    # Temperature related parameters
    microbial_degradation_rate_max_T_ref: float = declare(default=20, unit="°C", unit_comment="", description="the reference temperature", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    microbial_degradation_rate_max_A: float = declare(default=0., unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    microbial_degradation_rate_max_B: float = declare(default=3.98, unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", 
                                        value_comment="", references="The value for B (Q10) has been fitted from the evolution of Vmax measured by Coody et al. (1986, SBB), who provided the evolution of the maximal uptake of glucose by soil microorganisms at 4, 12 and 25 degree C.", DOI="",
                                       min_value="", max_value="", variable_type="parametyer", by="model_soil", state_variable_type="", edit_by="user")
    microbial_degradation_rate_max_C: float = declare(default=1, unit="adim", unit_comment="", description="parameter C (either 0 or 1)", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    # Water-related parameters
    water_volumic_mass: float = declare(default=1e6, unit="g.m-3", unit_comment="", description="Constant water volumic mass", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="extensive", edit_by="user")
    g_acceleration: float = declare(default=9.806, unit="m.s-2", unit_comment="", description="gravitationnal acceleration constant", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="extensive", edit_by="user")
    saturated_hydraulic_conductivity: float = declare(default=0.24, unit="adim", unit_comment="m.day-1", description="staturated hydraulic conductivity parameter", 
                                        value_comment="", references="clay loam estimated with Hydrus, bulk density = 1.42", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    theta_R: float = declare(default=0.0835, unit="adim", unit_comment="m3.m-3", description="Soil retention moisture", 
                                        value_comment="", references="clay loam estimated with Hydrus, bulk density = 1.42", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    theta_S: float = declare(default=0.4383, unit="adim", unit_comment="m3.m-3", description="Soil saturation moisture", 
                                        value_comment="", references="clay loam estimated with Hydrus, bulk density = 1.42", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    water_alpha: float = declare(default=0.0138, unit="cm-3", unit_comment="", description="alpha is the inverse of the air-entry value (or bubbling pressure)", 
                                        value_comment="", references="clay loam estimated with Hydrus, bulk density = 1.42", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    water_n: float = declare(default=1.3945, unit="cm-3", unit_comment="", description="alpha is the inverse of the air-entry value (or bubbling pressure)", 
                                        value_comment="", references="clay loam estimated with Hydrus, bulk density = 1.42", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    field_capacity: float = declare(default=0.36, unit="adim", unit_comment="", description="Soil moisture at which soil doesn't retain water anymore.", 
                                        value_comment="", references="Cornell university, case of sandy loam soil", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    permanent_wilting_point: float = declare(default=0.065, unit="adim", unit_comment="", description="Soil moisture at which soil doesn't retain water anymore.", 
                                        value_comment="", references="Cornell university, case of sandy loam soil", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    water_dt: float = declare(default=3600, unit="s", unit_comment="", description="Initialized time_step to try converging the soil water potential profile", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    min_water_dt: float = declare(default=10, unit="s", unit_comment="", description="min value for adaptative time-step in the convergence cycle for water potential profile", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    max_water_dt: float = declare(default=3600, unit="s", unit_comment="", description="max value for adaptative time-step in the convergence cycle for water potential profile", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    water_potential_tolerance: float = declare(default=1, unit="Pa", unit_comment="", description="tolerance for soil water potential gradient profile convergence", 
                                        value_comment="estimated from general usual for pressure head expression (1e-4 m) * rho * g_acceleration = 0.91 Pa", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    max_iterations: int = declare(default=20, unit="adim", unit_comment="", description="Maximal convergence cycle for water potential profile", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    C_solutes_background: float = declare(default=0, unit="mol.m-3", unit_comment="", description="Background non C and non N solutes concentration in soil", 
                                        value_comment="Raw estimation to align with inorganic N range for now", references="TODO", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    # W patch initialization parameters
    water_moisture_patch: float = declare(default=0.2, unit="mol.m-3", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in a located patch in soil", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_depth_water_moisture: float = declare(default=0., unit="m", unit_comment="", description="Depth of a nitrate patch in soil", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_uniform_width_water_moisture: float = declare(default=2*0.1, unit="m", unit_comment="", description="Width of the zone of the patch with uniform concentration of nitrate", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_transition_water_moisture: float = declare(default=1e-3, unit="m", unit_comment="", description="Variance of the normal law smooting the boundary transition of a nitrate patch with the background concentration", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")


    # N patch initialization parameters
    
    dissolved_mineral_N_patch: float = declare(default=20e-6, unit="mol.g-1", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in a located patch in soil", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_depth_mineralN: float = declare(default=10e-2, unit="m", unit_comment="", description="Depth of a nitrate patch in soil", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_uniform_width_mineralN: float = declare(default=4e-2, unit="m", unit_comment="", description="Width of the zone of the patch with uniform concentration of nitrate", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_transition_mineralN: float = declare(default=1e-3, unit="m", unit_comment="", description="Variance of the normal law smooting the boundary transition of a nitrate patch with the background concentration", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    # Temperature
    process_at_T_ref: float = declare(default=1., unit="adim", unit_comment="", description="Proportion of maximal process intensity occuring at T_ref", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    # hexose_degradation_rate_max_T_ref: float = declare(default=20, unit="°C", unit_comment="", description="the reference temperature", 
    #                                     value_comment="", references="", DOI="",
    #                                    min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    # hexose_degradation_rate_max_A: float = declare(default=0., unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", 
    #                                     value_comment="", references="", DOI="",
    #                                    min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    # hexose_degradation_rate_max_B: float = declare(default=3.98, unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", 
    #                                     value_comment="", references="The value for B (Q10) has been fitted from the evolution of Vmax measured by Coody et al. (1986, SBB), who provided the evolution of the maximal uptake of glucose by soil microorganisms at 4, 12 and 25 degree C.", DOI="",
    #                                    min_value="", max_value="", variable_type="parametyer", by="model_soil", state_variable_type="", edit_by="user")
    # hexose_degradation_rate_max_C: float = declare(default=1, unit="adim", unit_comment="", description="parameter C (either 0 or 1)", 
    #                                     value_comment="", references="", DOI="",
    #                                    min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    mucilage_degradation_rate_max_T_ref: float = declare(default=20, unit="°C", unit_comment="", description="the reference temperature", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    mucilage_degradation_rate_max_A: float = declare(default=0., unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    mucilage_degradation_rate_max_B: float = declare(default=3.98, unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parametyer", by="model_soil", state_variable_type="", edit_by="user")
    mucilage_degradation_rate_max_C: float = declare(default=1, unit="adim", unit_comment="", description="parameter C (either 0 or 1)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    cells_degradation_rate_max_T_ref: float = declare(default=20, unit="°C", unit_comment="", description="the reference temperature", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max_A: float = declare(default=0., unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max_B: float = declare(default=3.98, unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parametyer", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max_C: float = declare(default=1, unit="adim", unit_comment="", description="parameter C (either 0 or 1)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    # Kinetic soil degradation parameters
    # hexose_degradation_rate_max: float = declare(default=277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * 1) * 10, unit="mol.m-2.s-1", unit_comment="of hexose", description="Maximum degradation rate of hexose in soil", 
    #                                     value_comment="", references="According to what Jones and Darrah (1996) suggested, we assume that this Km is 2 times lower than the Km corresponding to root uptake of hexose (350 uM against 800 uM in the original article).", DOI="",
    #                                    min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    # Km_hexose_degradation: float = declare(default=1000 * 1e-6 / 12., unit="mol.g-1", unit_comment="of hexose", description="Affinity constant for soil hexose degradation", 
    #                                     value_comment="", references="We assume that the maximum degradation rate is 10 times higher than the maximum hexose uptake rate by roots", DOI="",
    #                                    min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    mucilage_degradation_rate_max: float = declare(default=277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * 1) * 10, unit="mol.m-2.s-1", unit_comment="of equivalent hexose", description="Maximum degradation rate of mucilage in soil", 
                                        value_comment="", references="We assume that the maximum degradation rate for mucilage is equivalent to the one defined for hexose.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    Km_mucilage_degradation: float = declare(default=1000 * 1e-6 / 12., unit="mol.g-1", unit_comment="of equivalent hexose", description="Affinity constant for soil mucilage degradation ", 
                                        value_comment="", references="We assume that Km for mucilage degradation is identical to the one for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max: float = declare(default=277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * 1) * 10 / 2, unit="mol.m-2.s-1", unit_comment="of equivalent hexose", description="Maximum degradation rate of root cells at the soil/root interface", 
                                        value_comment="", references="We assume that the maximum degradation rate for cells is equivalent to the half of the one defined for hexose.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    Km_cells_degradation: float = declare(default=1000 * 1e-6 / 12., unit="mol.g-1", unit_comment="of equivalent hexose", description="Affinity constant for soil cells degradation", 
                                        value_comment="", references="We assume that Km for cells degradation is identical to the one for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    
    

    def __init__(self, time_step, scene_xrange=1., scene_yrange=1., soil_depth=1., **scenario):
        """
        DESCRIPTION
        -----------
        __init__ method

        :param g: the root MTG
        :param time_step_in_seconds: time step of the simulation (s)
        :param scenario: mapping of existing variable initialization and parameters to superimpose.
        :return:
        """

        self.apply_scenario(**scenario)
        self.initiate_voxel_soil(scene_xrange, scene_yrange, soil_depth)
        self.time_step = time_step
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step, data=self.voxels, compartment="soil") 
        self.voxel_neighbor = {}

    # SERVICE FUNCTIONS

    # Just ressource for now
    def initiate_voxel_soil(self, scene_xrange=1., scene_yrange=1., soil_depth=1., 
                            voxel_length=3e-2, voxel_height=3e-2):
        """
        Note : not tested for now, just computed to support discussions.
        """
        self.voxels = {}

        
        self.planting_depth = 5e-2

        voxel_width = voxel_length
        voxel_volume = voxel_height * voxel_width * voxel_width

        self.delta_z = voxel_height
        self.voxels_Z_section_area = voxel_width * voxel_width
        
        self.voxel_number_x = int(scene_xrange / voxel_width) + 1
        actual_voxel_width = scene_xrange / self.voxel_number_x
        self.scene_xrange = scene_xrange

        self.voxel_number_y = int(scene_yrange / voxel_width) + 1
        actual_voxel_length = scene_yrange / self.voxel_number_y
        self.scene_yrange = scene_yrange

        voxel_volume = voxel_height * actual_voxel_width * actual_voxel_length

        scene_zrange = soil_depth
        self.voxel_number_z = int(scene_zrange / voxel_height) + 1
        self.scene_zrange = scene_zrange

        # Uncentered, positive grid
        y, z, x = np.indices((self.voxel_number_y, self.voxel_number_z, self.voxel_number_x))
        self.voxels["x1"] = x * actual_voxel_width
        self.voxels["x2"] = self.voxels["x1"] + actual_voxel_width
        self.voxels["y1"] = y * actual_voxel_length
        self.voxels["y2"] = self.voxels["y1"] + actual_voxel_length
        self.voxels["z1"] = z * voxel_height
        self.voxels["z2"] = self.voxels["z1"] + voxel_height

        self.voxel_dx = actual_voxel_width
        self.voxel_dy = actual_voxel_length
        self.voxel_dz = voxel_height

        self.voxel_grid_to_self("voxel_volume", voxel_volume)

        for name in self.state_variables + self.inputs:
            if name != "voxel_volume":
                self.voxel_grid_to_self(name, init_value=getattr(self, name))
        
        # Initialize volumic concentrations
        self.voxels["dry_soil_mass"] = 1e6 * self.voxels["voxel_volume"] * self.voxels["bulk_density"]
        self.voxels["water_volume"] = self.voxels["voxel_volume"] * self.voxels["soil_moisture"]
        self.voxels["C_mineralN_soil"] = self.voxels["dissolved_mineral_N"] * self.voxels["dry_soil_mass"] / self.voxels["water_volume"] / 14
        self.voxels["C_amino_acids_soil"] = self.voxels["DON"] * self.voxels["dry_soil_mass"] / self.voxels["water_volume"] / 14
        self.voxels["C_hexose_soil"] = self.voxels["DOC"] * self.voxels["dry_soil_mass"] / self.voxels["water_volume"] / 6 / 12
        self.voxels["Cv_solutes_soil"] = self.voxels["C_mineralN_soil"] # Until we are sure of proper initialization and balance of these different concentrations


    def voxel_grid_to_self(self, name, init_value):
        self.voxels[name] = np.zeros((self.voxel_number_y, self.voxel_number_z, self.voxel_number_x))
        self.voxels[name].fill(init_value)
        #setattr(self, name, self.voxels[name])
    

    def compute_mtg_voxel_neighbors(self, props):

        # necessary to get updated coordinates.
        # if "angle_down" in g.properties().keys():
        #     plot_mtg(g)

        for vid in props["vertex_index"].keys():
            if (vid not in props["voxel_neighbor"].keys()) or (props["voxel_neighbor"][vid] is None) or (props["length"][vid] > props["initial_length"][vid]):
                baricenter = (np.mean((props["x1"][vid], props["x2"][vid])) % self.scene_xrange, # min value is 0
                            np.mean((props["y1"][vid], props["y2"][vid])) % self.scene_yrange, # min value is 0
                            -np.mean((props["z1"][vid], props["z2"][vid])))
                testx1 = self.voxels["x1"] <= baricenter[0]
                testx2 = baricenter[0] <= self.voxels["x2"]
                testy1 = self.voxels["y1"] <= baricenter[1]
                testy2 = baricenter[1] <= self.voxels["y2"]
                testz1 = self.voxels["z1"] <= baricenter[2]
                testz2 = baricenter[2] <= self.voxels["z2"]
                test = testx1 * testx2 * testy1 * testy2 * testz1 * testz2
                try:
                    props["voxel_neighbor"][vid] = [int(v) for v in np.where(test)]
                except:
                    print(" WARNING, issue in computing the voxel neighbor for vid ", vid)
                    props["voxel_neighbor"][vid] = None
        
        return props
    
    
    def compute_mtg_voxel_neighbors_fast(self, data, hs, mask,
                                         xmin=0, ymin=0, zmin=0,
                                        periodic_xy=True, flip_z=False):
        
        # barycenters (vectorized)
        bx = 0.5 * (data[hs["x1"]] + data[hs["x2"]])
        by = 0.5 * (data[hs["y1"]] + data[hs["y2"]])
        bz = 0.5 * (data[hs["z1"]] + data[hs["z2"]])

        # The integer indices of the vertices where the boolean mask need is True
        # idx = np.nonzero(mask)[0]
        # xs, ys, zs = bx[idx], by[idx], bz[idx]
        xs, ys, zs = bx[mask], by[mask], bz[mask]

        # grid params
        Ny, Nz, Nx = self.voxel_number_y, self.voxel_number_z, self.voxel_number_x
        dx, dy, dz = self.voxel_dx, self.voxel_dy, self.voxel_dz

        if flip_z:
            zs = -zs
        
        if periodic_xy:
            Lx, Ly = Nx * dx, Ny * dy
            xs = (xs - xmin) % Lx + xmin
            ys = (ys - ymin) % Ly + ymin

        ix = np.floor((xs - xmin) / dx).astype(np.int32)
        iy = np.floor((ys - ymin) / dy).astype(np.int32)
        iz = np.floor((zs - zmin) / dz).astype(np.int32)

        # clamp to valid range (if not periodic or due to tiny FP drift)
        np.clip(ix, 0, Nx - 1, out=ix)
        np.clip(iy, 0, Ny - 1, out=iy)
        np.clip(iz, 0, Nz - 1, out=iz)

        return iy, iz, ix


    def apply_to_voxel(self, props):
        """
        This function computes the flow perceived by voxels surrounding the considered root segment.
        Note : not tested for now, just computed to support discussions.

        :param element: the considered root element.
        :param root_flows: The root flows to be perceived by soil voxels. The underlying assumptions are that only flows, i.e. extensive variables are passed as arguments.
        :return:
        """

        for name in self.inputs:
            self.voxels[name].fill(0)
        
        for vid in props["vertex_index"].keys():
            if props["length"][vid] > 0:
                vy, vz, vx = props["voxel_neighbor"][vid]
                for name in self.inputs:
                    # print(name,  props[name])
                    self.voxels[name][vy][vz][vx] += props[name][vid]


    def get_from_voxel(self, props, soil_outputs):
        """
        This function computes the soil states from voxels perceived by the considered root segment.
        Note : not tested for now, just computed to support discussions.

        :param element: the considered root element.
        :param soil_states: The soil states to be perceived by soil voxels. The underlying assumptions are that only intensive extensive variables are passed as arguments.
        :return:
        """
        for vid in props["vertex_index"].keys():
            vy, vz, vx = props["voxel_neighbor"][vid]
            for name in soil_outputs:
                if name != "voxel_neighbor":
                    props[name][vid] = self.voxels[name][vy][vz][vx]
        
        return props


    def pull_available_inputs(self, props):
        # vertices = props["vertex_index"].keys()
        vertices = [vid for vid in props["vertex_index"].keys() if props["living_struct_mass"][vid] > 0]
        
        for input, source_variables in self.pullable_inputs[props["model_name"]].items():
            if input not in props:
                props[input] = {}
            # print(input, source_variables)
            props[input].update({vid: sum([props[variable][vid]*unit_conversion 
                                           for variable, unit_conversion in source_variables.items()]) 
                                 for vid in vertices})
        return props

    
    def __call__(self, queue_plants_to_soil, queues_soil_to_plants, soil_outputs: list=[], *args):

        # We get fluxes and voxel interception from the plant mtgs (If none passed, soil model can be autonomous)
        # Waiting for all plants to put their outputs

        batch = []
        for _ in range(len(queues_soil_to_plants)):
            plant_data = queue_plants_to_soil.get()
            batch.append(plant_data)

            # Deplaced here because if one plant hangs, their is not reason not to retreive others in the meantime
            self.get_from_plant(plant_data)
        

        # Run the soil model
        self.choregrapher(module_family=self.__class__.__name__, *args)

        homogeneize_properties = True
        if homogeneize_properties:
            v = self.voxels
            v["dissolved_mineral_N"] = np.ones_like(v["dissolved_mineral_N"]) * (v["dissolved_mineral_N"] * (v["dry_soil_mass"])).sum() / (v["dry_soil_mass"]).sum()
            v["C_mineralN_soil"] = v["dissolved_mineral_N"] * (v["dry_soil_mass"] / (v["soil_moisture"] * v["voxel_volume"])) / 14 

        for plant_data in batch:
            plant_id = plant_data["plant_id"]

            self.send_to_plant(plant_data, soil_outputs)
            
            # Send a message to plants so that they can resume with last soil state
            queues_soil_to_plants[plant_id].put("finished")
        

    def get_from_plant(self, plant_data):
        """
        TODO : probably transfer to composite
        """
        # Unpacking message
        plant_id = plant_data["plant_id"]
        model_name = plant_data["model_name"]
        shm = SharedMemory(name=plant_id)
        buf = np.ndarray((35, 20000), dtype=np.float64, buffer=shm.buf)
        hs = plant_data["handshake"]
        vertices_mask = buf[hs["vertex_index"]] >= 1 # WARNING: convention

        iy, iz, ix = self.compute_mtg_voxel_neighbors_fast(buf, hs, mask=vertices_mask, flip_z=True)
        self.apply_to_voxel_fast(iy, iz, ix, buf, hs, model_name, vertices_mask)
        # Stored for sending to plants later
        self.voxel_neighbor[plant_id] = (iy, iz, ix)

        shm.close()


    def send_to_plant(self, plant_data, soil_outputs):
        """
        TODO : probably transfer to composite
        """
        # Then apply the states to the plants
        plant_id = plant_data["plant_id"]
        hs = plant_data["handshake"]
        shm = SharedMemory(name=plant_id)
        buf = np.ndarray((35, 20000), dtype=np.float64, buffer=shm.buf)
        vertices_mask = buf[hs["vertex_index"]] >= 1 # WARNING: convention
        self.get_from_voxel_fast(*self.voxel_neighbor[plant_id], buf, hs, soil_outputs, vertices_mask)

        shm.close()


    def apply_to_voxel_fast(self, iy, iz, ix, data, hs, model_name, mask):
        for name in self.inputs:
            self.voxels[name].fill(0.)
            
            if name in self.pullable_inputs[model_name]:
                source_variables = self.pullable_inputs[model_name][name]
                to_apply = np.zeros(mask.sum(), dtype=np.float64)
                for variable, unit_conversion in source_variables.items():
                    to_apply += unit_conversion * data[hs[variable]][mask]
            else:
                to_apply = data[hs[name]][mask]

            np.add.at(self.voxels[name], (iy, iz, ix), to_apply) 

            # print(name, self.voxels[name].sum())


    def get_from_voxel_fast(self, iy, iz, ix, data, hs, soil_outputs, mask):
        # cols = np.flatnonzero(mask)
        # Nx, Nz = self.voxel_number_x, self.voxel_number_z
        # linear_index = ((iy * Nz + iz) * Nx + ix)
        # print("idx shape", ix.shape)
        # print(linear_index.shape)
        for name in soil_outputs:
            # print('vs assigned', self.voxels[name].ravel())
            # print(name, ix[cols].shape, data[hs[name], mask].shape, self.voxels[name][iy, iz, ix].shape)
            data[hs[name], mask] = self.voxels[name][iy, iz, ix]
            # print(name, data[hs[name], :])

    
    # MODEL EQUATIONS
    
    def _soil_moisture(self, water_potential_soil):
        m = 1 - (1/self.water_n)
        return self.theta_R + (self.theta_S - self.theta_R) / (1 + np.abs(self.water_alpha * water_potential_soil)**self.water_n) ** m

    @segmentation
    @state
    def _C_mineralN_soil(self, dissolved_mineral_N, dry_soil_mass, soil_moisture, voxel_volume):
        return dissolved_mineral_N * (dry_soil_mass / (soil_moisture * voxel_volume)) / 14

    #TP@segmentation
    #TP@state
    def _C_amino_acids_soil(self, DOC, dry_soil_mass, soil_moisture, voxel_volume):
        return DOC * (dry_soil_mass * (soil_moisture / voxel_volume)) / 12 / self.ratio_C_per_amino_acid
    
    @postsegmentation
    @state
    def _Cv_solutes_soil(self, C_mineralN_soil, C_amino_acids_soil):
        return C_mineralN_soil + C_amino_acids_soil
    
    @state
    def _water_volume(self, soil_moisture, voxel_volume):
        return soil_moisture * voxel_volume
    
    @state
    def _water_potential_soil(self, voxel_volume, water_volume):
        """
        Water retention curve from van Genuchten 1980
        """
        m = 1 - (1/self.water_n)
        return - (1 / self.water_alpha) * (
                                            ((self.theta_S - self.theta_R) / ((water_volume / voxel_volume) - self.theta_R)) ** (1 / m) - 1 
                                        )** (1 / self.water_n)