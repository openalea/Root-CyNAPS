#  -*- coding: utf-8 -*-

"""
    Extract from openalea.rhizodep.model_growth
    ~~~~~~~~~~~~~

    The module :mod:`openalea.rhizodep.model_growth` defines the equations of root architectured growth.

    :copyright: see AUTHORS.
    :license: see LICENSE for details.
"""

import numpy as np
from math import pi
from dataclasses import dataclass

from openalea.mtg import *
from openalea.mtg.traversal import post_order, pre_order2, post_order2

from openalea.metafspm.component import Model, declare
from openalea.metafspm.component_factory import *

debug = False


@dataclass
class StaticRootGrowthModel(Model):
    """
    DESCRIPTION
    -----------
    Root growth model originating from Rhizodep shoot.py
    """

    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---
    # FROM SOIL MODEL
    soil_temperature: float = declare(default=15, unit="°C", unit_comment="", description="soil temperature in contact with roots", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_soil", state_variable_type="", edit_by="user")

    # FROM ANATOMY MODEL
    root_tissue_density: float = declare(default=0.10 * 1e6, unit="g.m-3", unit_comment="of structural mass", description="root_tissue_density", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")

    # --- INITIALIZE MODEL STATE VARIABLES ---
    type: str = declare(default="Normal_root_after_emergence", unit="", unit_comment="", description="Example segment type provided by root growth model", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="descriptor", edit_by="user")
    label: str = declare(default="Apex", unit="", unit_comment="", description="Example segment label provided by root growth model", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="descriptor", edit_by="user")
    root_order: int = declare(default=1, unit="", unit_comment="", description="Example root segment's axis order computed by the initiate_mtg method or provided as input", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="descriptor", edit_by="user")
    vertex_index: int = declare(default=1, unit="mol.s-1", unit_comment="", description="Unique vertex identifier stored for ease of value access", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="descriptor", edit_by="user")
    axis_index: str = declare(default="seminal_1", unit="dimensionless", unit_comment="", description="Unique axis identifier stored for ease of value access", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="descriptor", edit_by="user")
    axis_type: str = declare(default="seminal", unit="", unit_comment="", description="Example axis type provided by root growth model", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="descriptor", edit_by="user")
    radius: float = declare(default=3.5e-4, unit="m", unit_comment="", description="Example root segment radius", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialIntensive", edit_by="user")
    z1: float = declare(default=0., unit="m", unit_comment="", description="Depth of the segment tip computed by plantGL, colar side", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialIntensive", edit_by="user")
    z2: float = declare(default=0., unit="m", unit_comment="", description="Depth of the segment tip computed by plantGL, apex side", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialIntensive", edit_by="user")
    length: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example root segment length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    initial_length: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example root segment length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    struct_mass: float = declare(default=1.35e-4, unit="g", unit_comment="", description="Example root segment structural mass", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    initial_struct_mass: float = declare(default=1.35e-4, unit="g", unit_comment="", description="Same as struct_mass but corresponds to the previous time step; it is intended to record the variation", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    initial_living_root_hairs_struct_mass: float = declare(default=0., unit="g", unit_comment="", description="Same as struct_mass but corresponds to the previous time step; it is intended to record the variation", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    living_root_hairs_struct_mass: float = declare(default=0., unit="g", unit_comment="", description="Example root segment living root hairs structural mass", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    living_struct_mass: float = declare(default=0., unit="g", unit_comment="", description="Sum of segment and root hair living struct mass", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    root_hair_length: float = declare(default=1.e-3, unit="m", unit_comment="", description="Example root hair length", 
                                                    min_value="", max_value="", value_comment="", references="According to the work of Gahoonia et al. (1997), the root hair maximal length for wheat and barley evolves between 0.5 and 1.3 mm.", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    total_root_hairs_number: float = declare(default=30 * (1.6e-4 / 3.5e-4) * 3.e-3 * 1e3, unit="adim", unit_comment="", description="Example root hairs number on segment external surface", 
                                                    min_value="", max_value="", value_comment="30 * (1.6e-4 / radius) * length * 1e3", references=" According to the work of Gahoonia et al. (1997), the root hair density is about 30 hairs per mm for winter wheat, for a root radius of about 0.16 mm.", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    hexose_consumption_by_growth: float = declare(default=0., unit="mol.s-1", unit_comment="", description="Hexose consumption rate by growth is coupled to a root growth model", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    resp_growth: float = declare(default=0., unit="mol.s-1", unit_comment="of C", description="respiration rate during growth", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    hexose_consumption_by_fungus: float = declare(default=0., unit="mol.s-1", unit_comment="", description="Hexose consumption rate by fungus", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user") 
    distance_from_tip: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example distance from tip", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialIntensive", edit_by="user")
    volume: float = declare(default=1e-9, unit="m3", unit_comment="", description="Initial volume of the collar element", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    struct_mass_produced: float = declare(default=0, unit="g", unit_comment="of dry weight", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    root_hairs_struct_mass_produced: float = declare(default=0, unit="g", unit_comment="of dry weight", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    thermal_time_since_emergence: float = declare(default=0, unit="s", unit_comment="", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialIntensive", edit_by="user")
    thermal_time_since_cells_formation: float = declare(default=0, unit="s", unit_comment="", description="Thermal time since tissue formation", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialIntensive", edit_by="user")
    actual_time_since_formation: float = declare(default=0, unit="s", unit_comment="", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialIntensive", edit_by="user")
    tissue_formation_time: float = declare(default=0, unit="day", unit_comment="", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="model_growth", state_variable_type="NonInertialIntensive", edit_by="user")
    
    # Total state variable
    total_living_struct_mass: float =          declare(default=0, unit="g", unit_comment="of dry weight", description="", 
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="plant_scale_state", by="model_growth", state_variable_type="", edit_by="user")
                                                    

    # --- INITIALIZES MODEL PARAMETERS ---
    # Segment initialization
    D_ini: float = declare(default=0.8e-3, unit="m", unit_comment="", description="Initial tip diameter of the primary root", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    root_hair_radius: float = declare(default=12 * 1e-6 /2., unit="m", unit_comment="", description="Average radius of root hair", 
                                                    min_value="", max_value="", value_comment="", references="According to the work of Gahoonia et al. (1997), the root hair diameter is relatively constant for different genotypes of wheat and barley, i.e. 12 microns", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    root_hairs_lifespan: float = declare(default=46 * (60. * 60.), unit="s", unit_comment="time equivalent at temperature of T_ref", description="Average lifespan of a root hair", 
                                                    min_value="", max_value="", value_comment="", references="According to the data from McElgunn and Harrison (1969), the lifespan of wheat root hairs is 40-55h, depending on the temperature. For a temperature of 20 degree Celsius, the linear regression from this data gives 46h.", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    root_hairs_density: float = declare(default=30 * 1e3 / (0.16 / 2. * 1e-3), unit=".m-2", unit_comment="number of hairs par meter of root per meter of root radius", description="Average density of root hairs", 
                                                    min_value="", max_value="", value_comment="", references="According to the data from McElgunn and Harrison (1969), the elongation rate for wheat root hairs is about 0.080 mm h-1.", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    root_hair_max_length: float = declare(default=1 * 1e-3, unit="m", unit_comment="", description="Average maximal length of a root hair", 
                                                    min_value="", max_value="", value_comment="", references="According to the work of Gahoonia et al. (1997), the root hair maximal length for wheat and barley evolves between 0.5 and 1.3 mm.", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    root_hairs_elongation_rate: float = declare(default=0.080 * 1e-3 / (60. * 60.) /(12 * 1e-6 /2.), unit=".s-1", unit_comment="in meter per second per meter of root radius", description="Average elongation rate of root hairs", 
                                                    min_value="", max_value="", value_comment="", references="According to the data from McElgunn and Harrison (1969), the elongation rate for wheat root hairs is about 0.080 mm h-1.", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    LDs: float = declare(default=4000. * (60. * 60. * 24.) * 1000 * 1e-6, unit="s.m-1..g-1.m-3", unit_comment="time equivalent at temperature of T_ref", description="Average lifespan of a root hair", 
                                                    min_value="", max_value="", value_comment="", references="5000 day mm-1 g-1 cm3 (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    ER: float = declare(default=0.2 / (60. * 60. * 24.), unit=".s-1", unit_comment="time equivalent at temperature of T_ref", description="Emission rate of adventitious roots", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    n_seminal_roots: int = declare(default=5, unit="adim", unit_comment="", description="Maximal number of roots emerging from the base (including primary and seminal roots)", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    n_adventitious_roots: int = declare(default=10, unit="adim", unit_comment="", description="Maximal number of roots emerging from the base (including primary and seminal roots)", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    random_choice: float = declare(default=8, unit="adim", unit_comment="", description="We set the random seed, so that the same simulation can be repeted with the same seed:", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    D_sem_to_D_ini_ratio: float = declare(default=0.95, unit="adim", unit_comment="", description="Proportionality coefficient between the tip diameter of a seminal root and D_ini", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    CVDD: float = declare(default=0.2, unit="adim", unit_comment="", description="Relative variation of the daughter root diameter", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    starting_time_for_adventitious_roots_emergence: float = declare(default=(60. * 60. * 24.) * 9., unit="s", unit_comment="time equivalent at temperature of T_ref", description="Time when adventitious roots start to successively emerge", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    D_adv_to_D_ini_ratio: float = declare(default=0.8, unit="adim", unit_comment="", description="Proportionality coefficient between the tip diameter of an adventitious root and D_ini ", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")

    # Temperature
    process_at_T_ref: float = declare(default=1., unit="adim", unit_comment="", description="Proportion of maximal process intensity occuring at T_ref", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    T_ref: float = declare(default=0., unit="°C", unit_comment="", description="the reference temperature", 
                                                    min_value="", max_value="", value_comment="", references="We assume that relative growth is 0 at T_ref=0 degree Celsius, and linearily increases to reach 1 at 20 degree.", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    A: float = declare(default=1/20, unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    B: float = declare(default=0, unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    C: float = declare(default=0, unit="adim", unit_comment="", description="parameter C (either 0 or 1)", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")

    # C supply for elongation
    growing_zone_factor: float = declare(default=21., unit="adim", unit_comment="", description="Proportionality factor between the radius and the length of the root apical zone in which C can sustain root elongation", 
                                                    min_value="", max_value="", value_comment="", references="According to illustrations by Kozlova et al. (2020), the length of the growing zone corresponding to the root cap, meristem and elongation zones is about 8 times the diameter of the tip.", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")

    # potential development
    emergence_delay: float = declare(default=3.27 * (60. * 60. * 24.), unit="s", unit_comment="time equivalent at temperature of T_ref", description="Delay of emergence of the primordium", 
                                                    min_value="", max_value="", value_comment="", references="emergence_delay = 3 days (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    EL: float = declare(default=6.5e-4, unit="s-1", unit_comment="meters of root per meter of radius per second equivalent to T_ref_growth", description="Slope of the elongation rate = f(tip diameter) ", 
                                                    min_value="", max_value="", value_comment="", references="EL = 5 mm mm-1 day-1 (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    Km_elongation: float = declare(default=1250 * 1e-6 / 6., unit="mol.g-1", unit_comment="of hexose", description="Affinity constant for root elongation", 
                                                    min_value="", max_value="", value_comment="", references="According to Barillot et al. (2016b): Km for root growth is 1250 umol C g-1 for sucrose. According to Gauthier et al (2020): Km for regulation of the RER by sucrose concentration in hz = 100-150 umol C g-1", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    relative_nodule_thickening_rate_max: float = declare(default=20. / 100. / (24. * 60. * 60.), unit="s-1", unit_comment="", description="Maximal rate of relative increase in nodule radius", 
                                                    min_value="", max_value="", value_comment="", references="We consider that the radius can't increase by more than 20% every day (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    Km_nodule_thickening: float = declare(default=1250 * 1e-6 / 6. * 100, unit="mol.g-1", unit_comment="of hexose", description="Affinity constant for nodule thickening", 
                                                    min_value="", max_value="", value_comment="Km_elongation * 100", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    nodule_max_radius: float = declare(default=0.8e-3 * 20., unit="m", unit_comment="", description="Maximal radius of nodule", 
                                                    min_value="", max_value="", value_comment="Dini * 10", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    SGC: float = declare(default=0.0, unit="adim", unit_comment="", description="Proportionality coefficient between the section area of the segment and the sum of distal section areas", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    relative_root_thickening_rate_max: float = declare(default=5. / 100. / (24. * 60. * 60.), unit="s-1", unit_comment="", description="Maximal rate of relative increase in root radius", 
                                                    min_value="", max_value="", value_comment="", references="We consider that the radius can't increase by more than 5% every day (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    C_hexose_min_for_elongation : float = declare(default=1e-5, unit="mol.g-1", unit_comment="in mol of hexose per g of structural mass", description="Treshold hexose concentration for elongation", 
                                                    min_value="", max_value="", value_comment="", references="?", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    C_hexose_min_for_thickening: float = declare(default=1e-5, unit="mol.g-1", unit_comment="in mol of hexose per g of structural mass", description="Treshold hexose concentration for thikening", 
                                                    min_value="", max_value="", value_comment="", references="?", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    Km_thickening: float = declare(default=1250 * 1e-6 / 6., unit="mol.g-1", unit_comment="of hexose", description="Affinity constant for root thickening", 
                                                    min_value="", max_value="", value_comment="Km_elongation", references="We assume that the Michaelis-Menten constant for thickening is the same as for root elongation. (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")

    # actual growth
    struct_mass_C_content: float = declare(default=0.44 / 12.01, unit="mol.g-1", unit_comment="of carbon", description="C content of structural mass", 
                                                    min_value="", max_value="", value_comment="", references="We assume that the structural mass contains 44% of C. (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    yield_growth: float = declare(default=0.8, unit="adim", unit_comment="mol of CO2 per mol of C used for structural mass", description="Growth yield", 
                                                    min_value="", max_value="", value_comment="", references="We use the range value (0.75-0.85) proposed by Thornley and Cannell (2000)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")

    # Segmentation and primordium formation
    segment_length: float = declare(default=3. / 1000., unit="m", unit_comment="", description="Length of a segment", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    nodule_formation_probability: float = declare(default=0.5, unit="m", unit_comment="", description="Probability (between 0 and 1) of nodule formation for each apex that elongates", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    Dmin: float = declare(default=0.122 / 1000., unit="m", unit_comment="", description="Minimal threshold tip diameter (i.e. the diameter of the finest observable roots)", 
                                                    min_value="", max_value="", value_comment="", references="Dmin=0.05 mm (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    RMD: float = declare(default=0.57, unit="adim", unit_comment="", description="Average ratio of the diameter of the daughter root to that of the mother root", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    IPD: float = declare(default=0.00474, unit="m", unit_comment="", description="Inter-primordia distance", 
                                                    min_value="", max_value="", value_comment="", references="IPD = 7.6 mm (??)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    new_root_tissue_density: float = declare(default=0.10 * 1e6, unit="g.m3", unit_comment="of structural mass", description="root_tissue_density", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")

    # Growth durations
    GDs: float = declare(default=800 * (60. * 60. * 24.) * 1000. ** 2., unit="s.m-2", unit_comment="time equivalent at temperature of T_ref", description="Coefficient of growth duration", 
                                                    min_value="", max_value="", value_comment="", references="Reference: GDs=930. day mm-2 (Pagès et Picon-Cochard, 2014)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    main_roots_growth_extender: float = declare(default=100., unit="s.s-1", unit_comment="", description="Coefficient of growth duration extension, by which the theoretical growth duration is multiplied for seminal and adventitious roots", 
                                                    min_value="", max_value="", value_comment="", references="Reference: GDs=400. day mm-2 ()", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    GD_highest: float = declare(default=60 * (60. * 60. * 24.), unit="s.m-2", unit_comment="time equivalent at temperature of T_ref", description="For seminal and adventitious roots, a longer growth duration is applied", 
                                                    min_value="", max_value="", value_comment="Expected growth duration of a seminal root", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    GD_high: float = declare(default=6 * (60. * 60. * 24.), unit="s.m-2", unit_comment="time equivalent at temperature of T_ref", description="The growth duration has a probability of [1-GD_prob_medium] to equal GD_high", 
                                                    min_value="", max_value="", value_comment="", references="Estimated the longest observed lateral wheat roots observed in rhizoboxes (Rees et al., unpublished)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    GD_medium: float = declare(default=0.70 * (60. * 60. * 24.), unit="s.m-2", unit_comment="time equivalent at temperature of T_ref", description="The growth duration has a probability of [GD_prob_medium - GD_prob_low] to equal GD_medium", 
                                                    min_value="", max_value="", value_comment="", references="Estimated from the medium lateral wheat roots observed in rhizoboxes (Rees et al., unpublished)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    GD_low: float = declare(default=0.25 * (60. * 60. * 24.), unit="s.m-2", unit_comment="time equivalent at temperature of T_ref", description="The growth duration has a probability of [GD_prob_low] to equal GD_low", 
                                                    min_value="", max_value="", value_comment="", references="Estimated from the shortest lateral wheat roots observed in rhizoboxes (Rees et al., unpublished)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    GD_by_frequency: bool = declare(default=False, unit="adim", unit_comment="", description="As an alternative to using a single value of growth duration depending on diameter, we offer the possibility to rather define the growth duration as a random choice between three values (low, medium and high), depending on their respective probability", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    GD_prob_low: float = declare(default=0.50, unit="adim", unit_comment="", description="Probability for low growth duration", 
                                                    min_value="", max_value="", value_comment="", references="Estimated from the shortest lateral wheat roots observed in rhizoboxes (Rees et al., unpublished)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    GD_prob_medium: float = declare(default=0.85, unit="adim", unit_comment="Probability for medium growth duration", description="Coefficient of growth duration", 
                                                    min_value="", max_value="", value_comment="", references="Estimated from the medium lateral wheat roots observed in rhizoboxes (Rees et al., unpublished)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")

    # --- USER ORIENTED PARAMETERS FOR SIMULATION ---
    # initiate MTG
    random: bool = declare(default=True, unit="adim", unit_comment="", description="Allow random processes in growth", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    simple_growth_duration: bool = declare(default=False, unit="adim", unit_comment="", description="Allow growth according to the original Archisimple model", 
                                                    min_value="", max_value="", value_comment="", references="(Pagès et al., 2014)", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    initial_segment_length: float = declare(default=1e-3, unit="m", unit_comment="", description="Initial segment length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    initial_apex_length: float = declare(default=1e-4, unit="m", unit_comment="", description="Initial apex length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    initial_C_hexose_root: float = declare(default=1e-4, unit="mol.g-1", unit_comment="", description="Initial hexose concentration of root segments", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    input_file_path: str = declare(default="inputs", unit="m", unit_comment="", description="Filepath for input files", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    forcing_seminal_roots_events: bool = declare(default=False, unit="m", unit_comment="", description="a Boolean expliciting if seminal root events should be forced", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    seminal_roots_events_file: str = declare(default="seminal_roots_inputs.csv", unit="m", unit_comment="", description="Filepath pointing to input table to plan seminal root emergence event", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    forcing_adventitious_roots_events: bool = declare(default=False, unit="m", unit_comment="", description="a Boolean expliciting if adventicious root events should be forced", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    adventitious_roots_events_file: str = declare(default="adventitious_roots_inputs.csv", unit="adim", unit_comment="", description="Filepath pointing to input table to plan adventitious root emergence event", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    radial_growth: bool = declare(default=True, unit="adim", unit_comment="", description="equivalent to a Boolean expliciting whether radial growth should be considered or not", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    nodules: bool = declare(default=False, unit="adim", unit_comment="", description="a Boolean expliciting whether nodules could be formed or not", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    root_order_limitation: bool = declare(default=False, unit="adim", unit_comment="", description="a Boolean expliciting whether lateral roots should be prevented above a certain root order", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")
    root_order_treshold: int = declare(default=2, unit="adim", unit_comment="", description="the root order above which new lateral roots cannot be formed", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_growth", state_variable_type="", edit_by="user")

    def __init__(self, g, time_step_in_seconds: int=3600, **scenario: dict):
        """
        DESCRIPTION
        -----------
        __init__ method similar to the RhizoDep model, transposed here to initialize variables on static architectures that would be requiered by other components

        :param time_step_in_seconds: time step of the simulation (s)
        :param scenario: mapping of existing variable initialization and parameters to superimpose.
        :return:
        """
        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)
        
        self.g = g

        self.props = self.g.properties()
        self.time_step_in_seconds = time_step_in_seconds
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step_in_seconds, data=self.props)
        self.vertices = self.g.vertices(scale=self.g.max_scale())
        self.link_self_to_mtg(ignore=list(self.props.keys())) # Here we need to ignore properties already created by self in initiate mtg, because the general rule is to superimpose for state variables

        # SPECIFIC HERE, Select real children for collar element (vid == 1).
        # This is mandatory for correct collar-to-tip Hagen-Poiseuille flow partitioning.
        self.collar_children, self.collar_skip = [], []
        for vid in self.vertices:
            children = self.g.children(vid)
            # if self.props["type"][vid] in ('Support_for_seminal_root', 'Support_for_adventitious_root') and children: # Alternative as these properties can be overridden during the simulation
            if self.props["label"][vid] == "Segment" and self.props["length"][vid] == 0 and children:
                self.collar_skip += [vid]
                # self.collar_children += [k for k in children if self.props["type"][k] not in ('Support_for_seminal_root', 'Support_for_adventitious_root')]
                self.collar_children += [k for k in children if not (self.props["label"][k] == "Segment" and self.props["length"][k] == 0)] # Alternative as these properties can be overridden during the simulation

        # TODO introduce an option instead of commenting!
        self.initiate_heterogeneous_variables()


    def initiate_heterogeneous_variables(self):
        # We cover all the vertices in the MTG:
        for vid in self.g.vertices_iter(scale=1):
            # n represents the vertex:
            n = self.g.node(vid)
            n.vertex_index = vid
            n.volume = self.volume_from_radius_and_length(n, n.radius, n.length)
            n.struct_mass = n.volume * self.new_root_tissue_density
            n.living_struct_mass = n.struct_mass + n.living_root_hairs_struct_mass

        algo.orders(self.g)
        self.update_distance_from_tip()
        # self.initiate_heterogeneous_struct_mass_production()
        self.post_growth_updating()


    def comute_mtg_axes_id(self):
        g = self.g
        props = self.props
        root = next(g.component_roots_at_scale_iter(g.root, scale=1))
        seminal_id = 1
        adventitious_id = 1
        lateral_id = 1
        
        processed_vids = []
        
        for v in post_order2(g, root):
            if v not in processed_vids:
                axis = g.Axis(v)
                insertion_id = g.parent(min(axis))

                if insertion_id:
                    parent = g.node(insertion_id)
                    if parent.type == "Support_for_seminal_root":
                        props["axis_index"].update({v: f"seminal_{seminal_id}" for v in axis})
                        seminal_id += 1
                    elif parent.type == "Support_for_adventitious_root":
                        props["axis_index"].update({v: f"adventitious_{adventitious_id}" for v in axis})
                        adventitious_id += 1
                    else:
                        if props["root_order"][min(axis)] > 1:
                            props["axis_index"].update({v: f"lateral_{lateral_id}" for v in axis})
                            lateral_id += 1
                        else:
                            print("Uncaptured exception on ", v)
                else:
                    # If parent is None we now this is the main seminal axis
                    props["axis_index"].update({v: f"seminal_{seminal_id}" for v in axis})
                    seminal_id += 1
                
                processed_vids += axis

    @postsegmentation
    @state
    def update_distance_from_tip(self):
        """
        The function "distance_from_tip" computes the distance (in meter) of a given vertex from the apex
        of the corresponding root axis in the MTG "g" based on the properties "length" of all vertices.
        Note that the dist-to-tip of an apex is defined as its length (and not as 0).
        :return: the MTG with an updated property 'distance_from_tip'
        """

        # We define "root" as the starting point of the loop below:
        root_gen = self.g.component_roots_at_scale_iter(self.g.root, scale=1)
        root = next(root_gen)

        # We travel in the MTG from the root tips to the base:
        for vid in post_order(self.g, root):
            # We define the current root element as n:
            n = self.g.node(vid)
            # We define its direct successor as son:
            son_id = self.g.Successor(vid)
            son = self.g.node(son_id)

            # We record the initial distance_from_tip as the "former" one (to be used by other functions):
            if hasattr(n, "distance_from_tip"):
                # Only if this is not the first time it is computed
                n.former_distance_from_tip = n.distance_from_tip

            # We try to get the value of distance_from_tip for the neighbouring root element located closer to the apex of the root:
            try:
                # We calculate the new distance from the tip by adding its length to the distance of the successor:
                n.distance_from_tip = son.distance_from_tip + n.length
            except:
                # If there is no successor because the element is an apex or a root nodule:
                # Then we simply define the distance to the tip as the length of the element:
                n.distance_from_tip = n.length

    def volume_from_radius_and_length(self, element, radius: float, length: float):
        """
        This function computes the volume (m3) of a root element
        based on the properties radius (m) and length (m) and possibly on its type.
        :param element: the investigated node of the MTG
        :param radius: radius of the element
        :param length: length of the element
        :return: the volume of the element
        """

        # If this is a regular root segment
        if element.type != "Root_nodule":
            # We consider the volume of a cylinder
            volume = pi * radius ** 2 * length
        else:
            # We consider the volume of a sphere:
            volume = 4 / 3. * pi * radius ** 3

        return volume
    

    def initiate_heterogeneous_struct_mass_production(self):
        """
        This function initializes the struct mass production zones related to elongation, root hair development, mother costs related to lateral elongation.
        It also initializes the root hair on existing architecture
        """
        # We define "root" as the starting point of the loop below:
        root_gen = self.g.component_roots_at_scale_iter(self.g.root, scale=1)
        root = next(root_gen)

        # We travel in the MTG from the root tips to the base:
        for vid in post_order(self.g, root):
            n = self.g.node(vid)

            # If at root tip, for every axis based on its radius we...
            if len(self.g.children(vid)) == 0:
                # ... Initialize a cumulative
                processed_length = 0.
                processed_root_hair_length = 0.
                # ... Compute the elongation zone length and the supposed uniform struct mass production along it
                elongation_zone_length = n.radius * self.growing_zone_factor                
                uniform_struct_mass_production = (np.pi * n.radius ** 2) * (self.EL * n.radius * n.root_tissue_density) / elongation_zone_length

                # .. Then handle the consumption by elongating laterals branched on this axis 
                emergence_period = n.radius * self.EL * self.emergence_delay
                lateral_sustaining_root_length = n.radius * self.growing_zone_factor # NOTE : it is the same length as the main root elongation zone, which is normal, as the relative rate of elongation is the same between the main root and the lateral root
                max_lateral_root_struct_mass_production = np.pi * (n.radius * self.RMD) ** 2 * (self.EL * n.radius * self.RMD) * n.root_tissue_density / lateral_sustaining_root_length
                k_sustaining_laterals = max_lateral_root_struct_mass_production / lateral_sustaining_root_length

            # IF WE ARE IN THE ELONGATION ZONE
            if processed_length < elongation_zone_length:
                # If this is the last segment being partly partly affected by elongation
                if processed_length + n.length > elongation_zone_length:
                    n.struct_mass_produced = uniform_struct_mass_production * (elongation_zone_length - processed_length)
                    
                    # Here we begin recording for root hair zone length and uniform struct mass production along as it relies on the local radius
                    root_hair_elongation_zone_length = (n.radius * self.EL) * (self.root_hair_max_length / (self.root_hairs_elongation_rate * self.root_hair_radius))
                    uniform_root_hair_stuct_mass_production = n.root_tissue_density * (
                        self.root_hair_radius ** 2 * np.pi * (self.root_hairs_elongation_rate * self.root_hair_radius)) * (
                            self.root_hairs_density * n.radius * root_hair_elongation_zone_length) / root_hair_elongation_zone_length
                    # Additionnaly we record the struct mass produced for root hair along with core segment structural mass
                    n.root_hairs_struct_mass_produced = uniform_root_hair_stuct_mass_production * (n.length + processed_length - elongation_zone_length)
                    n.total_root_hairs_number = self.root_hairs_density * n.radius * (n.length + processed_length - elongation_zone_length)
                    processed_root_hair_length = n.length + processed_length - elongation_zone_length
                    n.root_hair_length = self.root_hair_max_length * (processed_root_hair_length / 2) / root_hair_elongation_zone_length

                # Otherwise it affects the whole segment length
                else:
                    n.struct_mass_produced = uniform_struct_mass_production * n.length

                    # IF WE ARE AT RECENT LATERAL BRANCHING ZONES (OPTION 2, SEE 1 BELLOW)
                    # parent = n.parent()
                    # if parent.order == n.order-1:
                    #     parent.struct_mass_produced = uniform_struct_mass_production * (elongation_zone_length - processed_length - n.length)

                # In any case, no root hair are formed in elongation zone
                n.total_root_hairs_number = 0.
                n.root_hair_length = 0.

            else:
                # IF WE ARE IN THE ROOT HAIR ELONGATION ZONE
                if processed_length < elongation_zone_length + root_hair_elongation_zone_length:
                    # If this is the last segment being partly partly affected by root_hair elongation
                    if processed_length + n.length > elongation_zone_length + root_hair_elongation_zone_length:
                        n.root_hairs_struct_mass_produced = uniform_root_hair_stuct_mass_production * (elongation_zone_length + root_hair_elongation_zone_length - processed_length)
                        
                        # Computing root hair average length on the remaining section of the segment
                        segment_growing_length = elongation_zone_length + root_hair_elongation_zone_length - processed_length
                        n.root_hair_length = (
                            segment_growing_length * self.root_hair_max_length * (processed_root_hair_length + (segment_growing_length / 2.)) / root_hair_elongation_zone_length + 
                            (n.length - segment_growing_length) * self.root_hair_max_length
                                                ) / n.length
                        
                    # Otherwise it affects the whole segment length
                    else:
                        n.root_hairs_struct_mass_produced = uniform_root_hair_stuct_mass_production * n.length
                        
                        # Computing the average root hair length on the whole segment
                        n.root_hair_length = self.root_hair_max_length * (processed_root_hair_length + (n.length / 2.)) / root_hair_elongation_zone_length
                        processed_root_hair_length += n.length
                
                else:
                    # In this case, root hair have reached their maximal length
                    n.root_hair_length = self.root_hair_max_length

                # In all case, the root hair density is the same
                n.total_root_hairs_number = self.root_hairs_density * n.radius * n.length
                    
                # IF WE ARE AT RECENT LATERAL BRANCHING ZONES (OPTION 1, commented because of wrong order numbering a collar with RSML inputs)
                if processed_length < emergence_period + elongation_zone_length:
                    n.struct_mass_produced = max(0., max_lateral_root_struct_mass_production - k_sustaining_laterals * ((n.distance_from_tip - emergence_period)%emergence_period))

            processed_length += n.length

    def post_growth_updating(self, modules_to_update=[], soil_boundaries_to_infer=[]):
        g = self.g
        props = self.props

        for module in modules_to_update:
            setattr(module, "vertices", self.vertices)

        # Select the base of the root
        root = next(g.component_roots_at_scale_iter(g.root, scale=1))

        props["total_living_struct_mass"][1] = 0

        debug = False
        if debug: print(self.step_elongating_elements, self.step_new_apices)

        # from root base to tips
        for vid in pre_order2(g, root):
            n = g.node(vid)
            n.living_struct_mass = n.struct_mass + n.living_root_hairs_struct_mass

            # We need to get the parent to compute mass partitionning.
            if vid in self.collar_children:
                parent = 1
            else:
                parent = g.parent(vid)
            p = g.node(parent)
            # We have to introduce this to get proper axis type
            graph_parent = g.node(g.parent(vid))

            if n.type == 'Base_of_the_root_system' or p is None:
                n.axis_type = 'seminal'
            else:
                if n.root_order == 1:
                    # First exception for pivot root that could be taken for a nodal otherwise
                    # (Given the structure of the first fake supporting elements)
                    if vid == max(self.collar_children):
                        n.axis_type = 'seminal'
                    elif n.type == 'Support_for_seminal_root' or n.type == 'Support_for_adventitious_root':
                        n.axis_type = 'seminal'
                    elif graph_parent.type == 'Support_for_seminal_root':
                        n.axis_type = 'seminal'
                    elif graph_parent.type == 'Support_for_adventitious_root':
                        n.axis_type = 'nodal'
                    elif graph_parent.axis_type == 'seminal':
                        n.axis_type = 'seminal'
                    elif graph_parent.axis_type == 'nodal':
                        n.axis_type = 'nodal'
                    else:
                        print('Uncaught exception')
                else:
                    n.axis_type = 'lateral'
                    
                if n.struct_mass > 0:
                    if vid not in self.collar_skip:
                        # For every vertex we update the considered living struct_mass for other modules.
                        props["total_living_struct_mass"][1] += n.living_struct_mass

                        if len(modules_to_update) > 0: # Only relevant if the growth model has been coupled, otherwise all properties have already been updated
                            if vid in self.step_new_apices and vid not in props["vertex_index"]:
                                
                                # We increment the vertex identifiers to be accesses in deficits
                                n.vertex_index = vid
                                
                                mass_fraction = n.living_struct_mass / (n.living_struct_mass + p.living_struct_mass)

                                for module in modules_to_update:
                                    for prop in module.massic_concentration:
                                        if mass_fraction > 0:
                                            initial_metabolite_amount = getattr(p, prop) * (p.initial_struct_mass + p.initial_living_root_hairs_struct_mass)
                                            props[prop].update({vid: initial_metabolite_amount * mass_fraction / n.living_struct_mass,
                                                                        parent: initial_metabolite_amount * (1-mass_fraction) / p.living_struct_mass})
                                        else:
                                            props[prop].update({vid: getattr(p, prop)})
                                            
                                    for prop in module.extensive_variables:
                                        initial_amount = getattr(p, prop)
                                        props[prop].update({vid: initial_amount * mass_fraction,
                                                                    parent: initial_amount * (1-mass_fraction)})
                                        
                                    for prop in module.descriptor:
                                        props[prop].update({vid: None})

                                # For soil related inputs, given new elements have been formed and we will not compute soil interception now, we infer the values from those of the parent
                                for prop in soil_boundaries_to_infer:
                                    setattr(n, prop, getattr(p, prop))


                            elif vid in self.step_elongating_elements:
                                # If this is an already emerged segment, it has its own dynamic regarless of parents
                                if n.initial_struct_mass > 0:
                                    for module in modules_to_update:
                                        for prop in module.massic_concentration:
                                            setattr(n, prop, getattr(n, prop)* (n.initial_struct_mass + n.initial_living_root_hairs_struct_mass) / n.living_struct_mass)

                                # Else if it elongated from a null structural mass, it shared ressources with its parent and we deal with it as for new apex creation
                                else:
                                    
                                    mass_fraction = n.living_struct_mass / (n.living_struct_mass + p.living_struct_mass)

                                    for module in modules_to_update:
                                        for prop in module.massic_concentration:
                                            initial_metabolite_amount = getattr(p, prop) * (p.initial_struct_mass + p.initial_living_root_hairs_struct_mass)
                                            props[prop].update({vid: initial_metabolite_amount * mass_fraction / n.living_struct_mass,
                                                                        parent: initial_metabolite_amount * (1-mass_fraction) / p.living_struct_mass})
                                                
                                        for prop in module.extensive_variables:
                                            initial_amount = getattr(p, prop)
                                            props[prop].update({vid: initial_amount * mass_fraction,
                                                                        parent: initial_amount * (1-mass_fraction)})
                                    
                                    if n.vertex_index is None:
                                        n.vertex_index = vid
                                    
                                    # If first elongation but primordia was formed earlier, needs access to soil states
                                    for prop in soil_boundaries_to_infer:
                                        setattr(n, prop, getattr(p, prop))
                            

                        #TODO remove, just for a figure
                        if n.label != "Apex":
                            if vid not in props["actual_time_since_formation"]:
                                n.actual_time_since_formation = 0
                            else:
                                n.actual_time_since_formation += self.time_step_in_seconds / 3600 / 24
                        else:
                            if vid not in props["actual_time_since_formation"]:
                                n.actual_time_since_formation = 0
                            non_meristem_length = n.radius * 2 * 1.3 #Kozlova et al. (2020), the length of the meristematic region 1.3 times the diameter of the tip
                            if non_meristem_length < n.length:
                                aging_length = n.length - non_meristem_length
                                n.actual_time_since_formation += (self.time_step_in_seconds / 3600 / 24) * aging_length / n.length

                        n.tissue_formation_time = n.thermal_time_since_cells_formation / 3600 / 24

        compute_axess_id = True
        if compute_axess_id:
            self.comute_mtg_axes_id()


    def __call__(self, *args, modules_to_update=[], soil_boundaries_to_infer=[]):
        super().__call__(*args)
        self.post_growth_updating(modules_to_update=modules_to_update, soil_boundaries_to_infer=soil_boundaries_to_infer)
    