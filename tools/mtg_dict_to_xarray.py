import pandas as pd
import xarray as xr
import pickle

# Global MTG metadata
description = "Rhizodep local root MTG properties over time (built from mtg dict properties)"

description_glob = "Rhizodep global root MTG properties over time"

# Coordinates in topology
# Note, the following coordinate (root_order + dist_to_ramif / (dist_to_ramif + distance_from_tip ) is not needed for computation
# It will thus only be added as coordinate during reading
# time is not included here because it results from xarray.merge
mtg_coordinates = dict(
    vid=dict(unit="adim", value_example=1, description="Root segment identifier index"),
    t=dict(unit="h", value_example=1, description="Model time step")
    # for now only t and vid coordinates have been chosen to match original mtg's properties' structure.
    # Using another topology coordinates as proposed above will need to be motivated by process regulation and code efficiency
    # Indeed, it needs to be parcimonious as storage size increase dramatically with too much dimensions
    # (filling unused space with nan) analysis dimensions rather should be built afterwards from xarray stack/unstack methods
)

# Properties of interest
# TODO : issues with RGB values (idea [0,1]?), list (idea computation from mtg topology?) and boolean properties (idea 0/1?)
props_metadata = dict(
    edge_type=dict(unit="adim", value_example="<", description="if '<': belongs to the same axis as the previous element; if '+': corresponds to the first element of a new axis"),
    label=dict(unit="adim", value_example="Segment", description="either Apex or Segment"),
    global_sucrose_deficit=dict(unit="adim", value_example="", description="not provided"),
    type=dict(unit="adim", value_example="Normal_root_after_emergence", description="Several possibilities, including Root_before_emergence (i.e. a primordium)"),
    root_order=dict(unit="adim", value_example=1, description="Root classes ordering according to successive branching events"),
    lateral_root_emergence_possibility=dict(unit="adim", value_example="Impossible", description="not provided"),
    emergence_cost=dict(unit="mol of hexose", value_example=0, description="not provided"),
    angle_down=dict(unit="°", value_example=8.94314236715159, description="not provided"),
    angle_roll=dict(unit="°", value_example=116.691298481059, description="not provided"),
    length=dict(unit="m", value_example=0.0035, description="Length of the root cylinder"),
    radius=dict(unit="m", value_example=0.00035, description="Radius of the root cylinder"),
    original_radius=dict(unit="m", value_example=0.00035, description="not provided"),
    initial_length=dict(unit="m", value_example=0.0035, description="not provided"),
    initial_radius=dict(unit="m", value_example=0.00035, description="not provided"),
    root_hair_radius=dict(unit="m", value_example=0.000006, description="not provided"),
    root_hair_length=dict(unit="m", value_example=0.001, description="not provided"),
    actual_length_with_hairs=dict(unit="m", value_example=0.0035, description="not provided"),
    living_root_hairs_number=dict(unit="adim", value_example=459.375, description="not provided"),
    dead_root_hairs_number=dict(unit="adim", value_example=0, description="not provided"),
    total_root_hairs_number=dict(unit="adim", value_example=459.375, description="not provided"),
    actual_time_since_root_hairs_emergence_started=dict(unit="s", value_example=263855, description="not provided"),
    thermal_time_since_root_hairs_emergence_started=dict(unit="s", value_example=131927, description="not provided"),
    actual_time_since_root_hairs_emergence_stopped=dict(unit="s", value_example=221967, description="not provided"),
    thermal_time_since_root_hairs_emergence_stopped=dict(unit="s", value_example=110983, description="not provided"),
    #all_root_hairs_formed=dict(unit="adim", value_example=True, description="not provided"),
    root_hairs_lifespan=dict(unit="s", value_example=165600, description="not provided"),
    root_hairs_external_surface=dict(unit="m2", value_example=1.7318E-05, description="not provided"),
    root_hairs_volume=dict(unit="m3", value_example=5.19541E-11, description="not provided"),
    living_root_hairs_external_surface=dict(unit="m2", value_example=1.7318E-05, description="External surface developed by all root hairs (excluding the surface of the main root cylinder)"),
    initial_living_root_hairs_external_surface=dict(unit="adim", value_example="not provided", description="not provided"),
    root_hairs_struct_mass=dict(unit="g", value_example=5.19541E-06, description="not provided"),
    root_hairs_struct_mass_produced=dict(unit="g", value_example=0, description="not provided"),
    living_root_hairs_struct_mass=dict(unit="g", value_example=5.19541E-06, description="not provided"),
    external_surface=dict(unit="m2", value_example=7.49286E-06, description="External surface developed by the main root cyclinder (excluding the surface of root hairs)"),
    initial_external_surface=dict(unit="m2", value_example=7.49286E-06, description="not provided"),
    volume=dict(unit="m3", value_example=1.34696E-09, description="not provided"),
    distance_from_tip=dict(unit="m", value_example=0.026998706, description="Distance between the root segment and the considered root axis tip"),
    former_distance_from_tip=dict(unit="m", value_example=0.026998706, description="Distance between the bottom of this element and the extremity of the root axis"),
    dist_to_ramif=dict(unit="m", value_example=0.00324, description="Distance between the root segment and the considered root axis tip"),
    actual_elongation=dict(unit="m", value_example="0", description="not provided"),
    actual_elongation_rate=dict(unit="m.s-1", value_example=0, description="not provided"),
    struct_mass=dict(unit="g", value_example=0.000134696, description="not provided"),
    initial_struct_mass=dict(unit="g", value_example=0.000134696, description="not provided"),
    initial_living_root_hairs_struct_mass=dict(unit="g", value_example=5.19541E-06, description="not provided"),
    C_sucrose_root=dict(unit="mol of sucrose.g-1 of root structural mass", value_example=0.066609287, description="Concentration of sucrose in the root (i.e. in the phloem vessels)"),
    C_hexose_root=dict(unit="mol of hexose.g-1 of root structural mass", value_example=0.128906873, description="Concentration of mobile hexose in the root (i.e. in the cortical and and epidermal symplasm)"),
    C_hexose_reserve=dict(unit="mol of hexose.g-1 of root structural mass", value_example=0.005, description="Concentration of immobile hexose in the root (i.e. in the cortical and and epidermal symplasm)"),
    C_hexose_soil=dict(unit="mol of hexose.g-1 of root structural mass", value_example=0.06751469, description="not provided"),
    Cs_mucilage_soil=dict(unit="mol of hexose.g-1 of root structural mass", value_example=2.4715E-05, description="not provided"),
    Cs_cells_soil=dict(unit="mol of hexose.g-1 of root structural mass", value_example=9.36124E-06, description="not provided"),
    sucrose_root=dict(unit="mol of sucrose", value_example="not provided", description="not provided"),
    hexose_root=dict(unit="mol of hexose", value_example="not provided", description="not provided"),
    hexose_reserve=dict(unit="mol of hexose", value_example="not provided", description="not provided"),
    hexose_soil=dict(unit="mol of hexose", value_example="not provided", description="not provided"),
    mucilage_soil=dict(unit="mol of hexose", value_example="not provided", description="not provided"),
    cells_soil=dict(unit="mol of hexose", value_example="not provided", description="not provided"),
    Deficit_sucrose_root=dict(unit="mol of sucrose.g-1 of root structural mass", value_example=0, description="not provided"),
    Deficit_hexose_root=dict(unit="mol of hexose.g-1 of root structural mass", value_example=0, description="not provided"),
    Deficit_hexose_reserve=dict(unit="mol of hexose.g-1 of root structural mass", value_example=0, description="not provided"),
    Deficit_hexose_soil=dict(unit="mol of hexose.g-1 of root structural mass", value_example=0, description="not provided"),
    Deficit_mucilage_soil=dict(unit="mol of hexose.g-1 of root structural mass", value_example=0, description="not provided"),
    Deficit_cells_soil=dict(unit="mol of hexose.g-1 of root structural mass", value_example=0, description="not provided"),
    Deficit_sucrose_root_rate=dict(unit="mol of sucrose.g-1 of root structural mass.s-1", value_example="not provided", description="not provided"),
    Deficit_hexose_root_rate=dict(unit="mol of hexose.g-1 of root structural mass.s-1", value_example="not provided", description="not provided"),
    Deficit_hexose_reserve_rate=dict(unit="mol of hexose.g-1 of root structural mass.s-1", value_example="not provided", description="not provided"),
    Deficit_hexose_soil_rate=dict(unit="mol of hexose.g-1 of root structural mass.s-1", value_example="not provided", description="not provided"),
    Deficit_mucilage_soil_rate=dict(unit="mol of hexose.g-1 of root structural mass.s-1", value_example="not provided", description="not provided"),
    Deficit_cells_soil_rate=dict(unit="mol of hexose.g-1 of root structural mass.s-1", value_example="not provided", description="not provided"),
    resp_maintenance=dict(unit="mol of CO2", value_example=1.89205E-09, description="Amount of CO2 emitted because of root maintenance processes"),
    resp_growth=dict(unit="mol of CO2", value_example=3.96239E-08, description="Amount of CO2 emitted because of root growth"),
    hexose_growth_demand=dict(unit="mol of hexose", value_example=0, description="not provided"),
    hexose_possibly_required_for_elongation=dict(unit="mol of hexose", value_example=0, description="not provided"),
    hexose_consumption_by_growth=dict(unit="mol of hexose", value_example=3.30199E-08, description="Amount of hexose used by root growth processes"),
    hexose_production_from_phloem=dict(unit="mol of hexose", value_example=2.8353E-08, description="Amount of hexose generated from phloem unloading and conversion of sucrose into hexose"),
    sucrose_loading_in_phloem=dict(unit="mol of sucrose", value_example=0, description="Amount of sucrose generated from phloem reloading"),
    hexose_mobilization_from_reserve=dict(unit="mol of hexose", value_example=1.61553E-07, description="Amount of hexose mobilized from the reserve pool"),
    hexose_immobilization_as_reserve=dict(unit="mol of hexose", value_example=1.61553E-07, description="Amount of hexose immobilized into the reserve pool"),
    hexose_exudation=dict(unit="mol of hexose", value_example=7.45467E-08, description="not provided"),
    hexose_uptake=dict(unit="mol of hexose", value_example=5.55861E-11, description="not provided"),
    mucilage_secretion=dict(unit="mol of hexose", value_example=3.11981E-11, description="not provided"),
    cells_release=dict(unit="mol of hexose", value_example=0, description="not provided"),
    total_rhizodeposition=dict(unit="mol of hexose", value_example=7.45223E-08, description="not provided"),
    hexose_degradation=dict(unit="mol of hexose", value_example=5.34181E-10, description="not provided"),
    mucilage_degradation=dict(unit="mol of hexose", value_example=3.29279E-11, description="not provided"),
    cells_degradation=dict(unit="mol of hexose", value_example=2.19778E-12, description="not provided"),
    specific_net_exudation=dict(unit="mol of hexose", value_example=0, description="not provided"),
    growth_duration=dict(unit="s", value_example=7310800, description="not provided"),
    life_duration=dict(unit="s", value_example=5282928, description="not provided"),
    actual_time_since_primordium_formation=dict(unit="s", value_example=428400, description="not provided"),
    actual_time_since_emergence=dict(unit="s", value_example=428400, description="not provided"),
    actual_time_since_growth_stopped=dict(unit="s", value_example=0, description="not provided"),
    actual_time_since_death=dict(unit="s", value_example=0, description="not provided"),
    thermal_time_since_primordium_formation=dict(unit="s", value_example=214200, description="not provided"),
    thermal_time_since_emergence=dict(unit="s", value_example=214200, description="Thermal time since root segment emerged"),
    thermal_potential_time_since_emergence=dict(unit="s", value_example=0, description="not provided"),
    thermal_time_since_growth_stopped=dict(unit="s", value_example=0, description="not provided"),
    thermal_time_since_death=dict(unit="s", value_example=0, description="not provided"),
    potential_length=dict(unit="m", value_example=0.0035, description="not provided"),
    theoretical_radius=dict(unit="m", value_example=0.00035, description="not provided"),
    potential_radius=dict(unit="m", value_example=0.00035, description="not provided"),
    struct_mass_produced=dict(unit="g", value_example=0, description="not provided"),
    actual_potential_time_since_emergence=dict(unit="s", value_example="not provided", description="not provided"),
    x1=dict(unit="m", value_example=-0.000120903, description="not provided"),
    y1=dict(unit="m", value_example=0, description="not provided"),
    z1=dict(unit="m", value_example=-0.007997911, description="not provided"),
    x2=dict(unit="m", value_example=-0.000526162, description="not provided"),
    y2=dict(unit="m", value_example=-0.000426887, description="not provided"),
    z2=dict(unit="m", value_example=-0.011448061, description="not provided"),
    # color=dict(unit="adim", value_example=[255, 0, 0], description="not provided"),
    struct_mass_contributing_to_elongation=dict(unit="g", value_example=0.000215513, description="not provided"),
    growing_zone_C_hexose_root=dict(unit="mol of hexose.g-1 of root structural mass.s-1", value_example=0.054552285, description="not provided"),
    # list_of_elongation_supporting_elements=dict(unit="adim", value_example=[5, 3], description="not provided"),
    # list_of_elongation_supporting_elements_hexose=dict(unit="adim", value_example=[4.974831108421144e-06, 6.781909480204513e-06], description="not provided"),
    # list_of_elongation_supporting_elements_mass=dict(unit="adim", value_example=[0.0001339703768358163, 8.154287920044353e-05], description="not provided"),
    hexose_available_for_thickening=dict(unit="mol of hexose", value_example=1.74398E-05, description="not provided"),
    phloem_surface=dict(unit="m2", value_example=7.49286E-06, description="not provided"),
    stelar_parenchyma_surface=dict(unit="m2", value_example="not provided", description="not provided"),
    cortical_parenchyma_surface=dict(unit="m2", value_example="not provided", description="not provided"),
    epidermis_surface_without_hairs=dict(unit="m2", value_example="not provided", description="not provided"),
    relative_conductance_walls=dict(unit="not provided", value_example="not provided", description="not provided"),
    relative_conductance_endodermis=dict(unit="not provided", value_example="not provided", description="not provided"),
    relative_conductance_exodermis=dict(unit="not provided", value_example="not provided", description="not provided"),
    hexose_consumption_by_growth_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    hexose_production_from_phloem_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    sucrose_loading_in_phloem_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    net_sucrose_unloading_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    hexose_immobilization_as_reserve_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    hexose_mobilization_from_reserve_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    net_hexose_immobilization_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    resp_maintenance_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    hexose_exudation_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    phloem_hexose_exudation_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    permeability_coeff=dict(unit="g.m-2.s-1", value_example=3.59228E-06, description="not provided"),
    hexose_uptake_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    phloem_hexose_uptake_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    soil_hexose_degradation_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    root_mucilage_secretion_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    soil_mucilage_degradation_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    root_cells_release_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    soil_cells_degradation_rate=dict(unit="mol of hexose.s-1", value_example="not provided", description="not provided"),
    phloem_hexose_exudation=dict(unit="mol of hexose", value_example="not provided", description="not provided"),
    phloem_hexose_uptake=dict(unit="mol of hexose", value_example="not provided", description="not provided"),
    net_hexose_exudation=dict(unit="mol of hexose", value_example=7.44911E-08, description="not provided"),
    biomass=dict(unit="g", value_example=0.002795425, description="not provided"),
    net_hexose_exudation_rate_per_day_per_gram=dict(unit="mol of hexose.day-1.g-1", value_example=0.038371437, description="not provided"),
    net_hexose_exudation_rate_per_day_per_cm=dict(unit="mol of hexose.day-1.cm-1", value_example=1.53366E-05, description="not provided"),
    # Next nitrogen properties
    Nm=dict(unit="mol N.g-1", value_example=float(1e-4), description="not provided"),
    AA=dict(unit="mol N.g-1", value_example=float(9e-4), description="not provided"),
    struct_protein=dict(unit="mol N.g-1", value_example=float(0), description="not provided"),
    storage_protein=dict(unit="mol N.g-1", value_example=float(0), description="not provided"),
    xylem_Nm=dict(unit="mol N.s-1", value_example=float(1e-4), description="not provided"),
    xylem_AA=dict(unit="mol N.s-1", value_example=float(1e-4), description="not provided"),
    phloem_AA=dict(unit="mol N.s-1", value_example=float(1e-4), description="not provided"),
    import_Nm=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    export_Nm=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    export_AA=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_Nm_soil=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_Nm_xylem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_Nm_soil_xylem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_AA_soil=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_AA_phloem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_AA_soil_xylem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    AA_synthesis=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    struct_synthesis=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    storage_synthesis=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    AA_catabolism=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    storage_catabolism=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    xylem_struct_mass=dict(unit="g", value_example=float(1e-3), description="not provided"),
    phloem_struct_mass = dict(unit="g", value_example=float(1e-3), description="not provided"),
    axial_advection_Nm_xylem = dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    axial_advection_AA_xylem = dict(unit="mol AA.s-1", value_example=float(0), description="not provided"),
    axial_diffusion_Nm_xylem = dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    axial_diffusion_AA_xylem = dict(unit="mol AA.s-1", value_example=float(0), description="not provided"),
    axial_diffusion_AA_phloem = dict(unit="mol AA.s-1", value_example=float(0), description="not provided"),
    # Water model
    xylem_water=dict(unit="mol H2O", value_example=float(0), description="not provided"),
    radial_import_water=dict(unit="mol H2O.s-1", value_example=float(0), description="not provided"),
    axial_export_water_up=dict(unit="mol H2O.s-1", value_example=float(0), description="not provided"),
    axial_import_water_down=dict(unit="mol H2P.s-1", value_example=float(0), description="not provided"),
    # Topology model
    root_exchange_surface=dict(unit="m2", value_example=float(0), description="not provided"),
    cylinder_exchange_surface=dict(unit="m2", value_example=float(0), description="not provided"),
    stele_exchange_surface=dict(unit="m2", value_example=float(0), description="not provided"),
    phloem_exchange_surface=dict(unit="m2", value_example=float(0), description="not provided"),
    apoplasmic_stele=dict(unit="adim", value_example=float(0.5), description="not provided"),
    xylem_volume=dict(unit="m3", value_example=float(0), description="not provided"),
    # Soil boundaries
    soil_water_pressure=dict(unit="Pa", value_example=float(-0.1e6), description="not provided"),
    soil_temperature=dict(unit="K", value_example=float(283.15), description="not provided"),
    soil_Nm=dict(unit="mol N.m-3", value_example=float(0.5), description="not provided"),
    soil_AA=dict(unit="mol AA.m-3", value_example=float(0), description="not provided"),
    # Total properties
    # States
    total_Nm=dict(unit="mol", value_example="not provided", description="not provided"),
    total_hexose=dict(unit="mol", value_example="not provided", description="not provided"),
    total_cytokinins=dict(unit="mol", value_example="not provided", description="not provided"),
    total_struct_mass=dict(unit="mol", value_example="not provided", description="not provided"),
    xylem_total_Nm=dict(unit="mol", value_example="not provided", description="not provided"),
    xylem_total_AA=dict(unit="mol", value_example="not provided", description="not provided"),
    phloem_total_AA=dict(unit="mol", value_example="not provided", description="not provided"),
    xylem_total_water=dict(unit="mol", value_example="not provided", description="not provided"),
    xylem_total_volume=dict(unit="m3", value_example="not provided", description="not provided"),
    xylem_total_pressure=dict(unit="Pa", value_example="not provided", description="not provided"),
    # Flows
    Nm_root_shoot_xylem=dict(unit="mol", value_example="not provided",  description="not provided"),
    AA_root_shoot_xylem=dict(unit="mol", value_example="not provided", description="not provided"),
    AA_root_shoot_phloem=dict(unit="mol", value_example="not provided", description="not provided"),
    cytokinins_root_shoot_xylem=dict(unit="mol.s-1", value_example="not provided", description="not provided"),
    cytokinin_synthesis=dict(unit="mol.s-1", value_example="not provided", description="not provided"),
    water_root_shoot_xylem=dict(unit="mol", value_example="not provided", description="not provided")
)


def mtg_to_dataset(mtg, variables, coordinates=mtg_coordinates, description=description, time=0):
    # convert dict to dataframe with index corresponding to coordinates in topology space
    # (not just x, y, z, t thanks to MTG structure)
    props_df = pd.DataFrame.from_dict(mtg.properties())
    props_df["vid"] = props_df.index
    props_df["t"] = [time for k in range(props_df.shape[0])]
    props_df = props_df.set_index(list(coordinates.keys()))

    # Select properties actually used in the current version of the target model
    props_df = props_df[list(variables.keys())]
    # df = df[list(variables.keys())].fillna(0)
    # Filter duplicated indexes
    props_df = props_df[~props_df.index.duplicated()]

    # Convert to xarray with given dimensions to spatialize selected properties
    props_ds = props_df.to_xarray()

    # Dataset global attributes
    props_ds.attrs["description"] = description

    # Dataset coordinates' attribute metadata
    for k, v in coordinates.items():
        getattr(props_ds, k).attrs.update(v)

    # Dataset variables' attribute metadata
    for k, v in variables.items():
        getattr(props_ds, k).attrs.update(v)

    return props_ds
