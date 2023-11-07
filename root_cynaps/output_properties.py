### Output parameters
state_extracts = dict(
    # Next nitrogen properties
    Nm=dict(unit="mol N.g-1", value_example=float(1e-4), description="not provided"),
    AA=dict(unit="mol N.g-1", value_example=float(9e-4), description="not provided"),
    struct_protein=dict(unit="mol N.g-1", value_example=float(0), description="not provided"),
    #storage_protein=dict(unit="mol N.g-1", value_example=float(0), description="not provided"),
    xylem_Nm=dict(unit="mol N.s-1", value_example=float(1e-4), description="not provided"),
    xylem_AA=dict(unit="mol N.s-1", value_example=float(1e-4), description="not provided"),
    #xylem_struct_mass=dict(unit="g", value_example=float(1e-3), description="not provided"),
    #phloem_struct_mass=dict(unit="g", value_example=float(1e-3), description="not provided"),
    # Water model
    xylem_water=dict(unit="mol H2O", value_example=float(0), description="not provided"),
    # Topology model
    #root_exchange_surface=dict(unit="m2", value_example=float(0), description="not provided"),
    #stele_exchange_surface=dict(unit="m2", value_example=float(0), description="not provided"),
    #phloem_exchange_surface=dict(unit="m2", value_example=float(0), description="not provided"),
    #apoplasmic_stele=dict(unit="adim", value_example=float(0.5), description="not provided"),
    #xylem_volume=dict(unit="m3", value_example=float(0), description="not provided"),
    # Soil boundaries
    #soil_water_pressure=dict(unit="Pa", value_example=float(-0.1e6), description="not provided"),
    #soil_temperature=dict(unit="K", value_example=float(283.15), description="not provided"),
    #soil_Nm=dict(unit="mol N.m-3", value_example=float(0.5), description="not provided"),
    #soil_AA=dict(unit="mol AA.m-3", value_example=float(0), description="not provided"),
    # Rhizodep properties
    struct_mass=dict(unit="g", value_example=0.000134696, description="not provided"),
    #distance_from_tip=dict(unit="m", value_example=float(0.026998706), description="Distance between the root segment and the considered root axis tip")
    # C_hexose_root=dict(unit="mol.g-1", value_example=0.000134696, description="not provided")
)

flow_extracts = dict(
    import_Nm=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    import_AA=dict(unit="mol AA.s-1", value_example=float(0), description="not provided"),
    export_Nm=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    export_AA=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_Nm_soil=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_Nm_xylem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_Nm_soil_xylem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_AA_soil=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_AA_phloem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    diffusion_AA_soil_xylem=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    displaced_Nm_in=dict(unit="mol N.time_step-1", value_example=float(0), description="not provided"),
    #displaced_Nm_out=dict(unit="mol N.time_step-1", value_example=float(0), description="not provided"),
    displaced_AA_in=dict(unit="mol N.time_step-1", value_example=float(0), description="not provided"),
    #displaced_AA_out=dict(unit="mol N.time_step-1", value_example=float(0), description="not provided"),
    cumulated_radial_exchanges_Nm=dict(unit="mol N.time_step-1", value_example=float(0), description="not provided"),
    cumulated_radial_exchanges_AA=dict(unit="mol N.time_step-1", value_example=float(0), description="not provided"),
    AA_synthesis=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    struct_synthesis=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    #storage_synthesis=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    AA_catabolism=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    #storage_catabolism=dict(unit="mol N.s-1", value_example=float(0), description="not provided"),
    # Water model
    radial_import_water=dict(unit="mol H2O.s-1", value_example=float(0), description="not provided"),
    #axial_export_water_up=dict(unit="mol H2O.h-1", value_example=float(0), description="not provided"),
    #axial_import_water_down=dict(unit="mol H2O.h-1", value_example=float(0), description="not provided"),
    shoot_uptake=dict(unit="mol H2O.h-1", value_example=float(0), description="not provided")
)

global_state_extracts = dict(
    total_Nm=dict(unit="mol", value_example="not provided",  description="not provided"),
    total_AA=dict(unit="mol", value_example="not provided", description="not provided"),
    total_hexose=dict(unit="mol", value_example="not provided", description="not provided"),
    total_cytokinins=dict(unit="mol", value_example="not provided", description="not provided"),
    total_struct_mass=dict(unit="mol", value_example="not provided", description="not provided"),
    xylem_total_Nm=dict(unit="mol", value_example="not provided", description="not provided"),
    xylem_total_AA=dict(unit="mol", value_example="not provided", description="not provided"),
    phloem_total_AA=dict(unit="mol", value_example="not provided", description="not provided"),
    xylem_total_water=dict(unit="mol", value_example="not provided", description="not provided"),
    xylem_total_pressure=dict(unit="Pa", value_example="not provided", description="not provided")
)

global_flow_extracts = dict(
    Nm_root_shoot_xylem=dict(unit="mol.time_step-1", value_example="not provided",  description="not provided"),
    AA_root_shoot_xylem=dict(unit="mol.time_step-1", value_example="not provided", description="not provided"),
    Unloading_Amino_Acids=dict(unit="mol.time_step-1", value_example="not provided", description="not provided"),
    Export_cytokinins=dict(unit="UA.time_step-1", value_example="not provided", description="not provided"),
    cytokinin_synthesis=dict(unit="mol", value_example="not provided", description="not provided"),
    actual_transpiration=dict(unit="mol.time_step-1", value_example="not provided", description="not provided"),
    Total_Transpiration=dict(unit="mol.time_step-1", value_example="not provided", description="not provided"),
    total_AA_rhizodeposition=dict(unit="mol.time_step-1", value_example="not provided", description="not provided")
)
