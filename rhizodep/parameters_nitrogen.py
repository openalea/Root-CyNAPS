"""
    rhizodep.parameters
    ~~~~~~~~~~~~~~~~~~~~~~

    The module defines the constant parameters for the :mod:`rhizodep.nitrogen` module.

"""
### External conditions parameters

init_soil_homogeneous = {
            "zmax_soil_Nm": float(              -0.02),
            "soil_Nm_variance": float(          0.0001),
            "soil_Nm_slope": float(             0),
            "scenario": int(                    0)
            }

init_soil_linear = {
            "zmax_soil_Nm": float(              -0.02),
            "soil_Nm_variance": float(          0.0001),
            "soil_Nm_slope": float(             25),
            "scenario": int(                    0)
            }

init_soil_patch = {
            "zmax_soil_Nm": float(              -0.02),
            "soil_Nm_variance": float(          0.0001),
            "soil_Nm_slope": float(             25),
            "scenario": int(                    1)
            }

### State variables initialisation

init_N = {
            "Nm": float(                        1e-4),
            "AA": float(                        1e-4),
            "influx_Nm": float(                 0),
            "diffusion_AA_soil": float(         0),
            "loading_Nm": float(                0),
            "loading_AA": float(                0),
            "diffusion_Nm_phloem": float(       0),
            "diffusion_AA_phloem": float(       0),
            "AA_synthesis": float(              0),
            "struct_synthesis": float(          0),
            "storage_synthesis": float(         0),
            "AA_catabolism": float(             0),
            "storage_catabolism": float(        0),
            "xylem_Nm": float(                  1e-4),
            "xylem_AA": float(                  1e-4),
            "xylem_struct_mass": float(         1e-3),
            "phloem_Nm": float(                 1e-4),
            "phloem_AA": float(                 1e-4),
            "phloem_struct_mass": float(        1e-3),
            "Nm_root_shoot_xylem": float(       0),
            "AA_root_shoot_xylem": float(       0),
            "Nm_root_shoot_phloem": float(      0),
            "AA_root_shoot_phloem": float(      0)
          }


### Model parameters

# Global parameters
xylem_to_root: float = 0.2
phloem_to_root: float = 0.15

transport_N = {
            # kinetic parameters
            "affinity_Nm_root": float(          1e-4),
            "vmax_Nm_emergence": float(         1e-9),
            "affinity_Nm_xylem": float(         1e-4),
            "vmax_AA_emergence": float(         1e-9),
            "affinity_AA_xylem": float(         1e-4),
            "diffusion_phloem": float(          1e-8),
            "diffusion_soil": float(            1e-9),
            # metabolism-related parameters
            "transport_C_regulation": float(    1e-2),
            "transport_N_regulation": float(    0.01),
            # architecture parameters
            "xylem_to_root": xylem_to_root,
            "phloem_to_root": phloem_to_root,
            "epiderm_differentiation": float(   1e-6),
            "endoderm_differentiation": float(  1e-6)
                }

metabolism_N = {
            # kinetic parameters
            "smax_AA": float(                   0),
            "affinity_Nm_AA": float(            0.001),
            "affinity_C_AA": float(             0.001),
            "smax_struct": float(               0),
            "affinity_AA_struct": float(        0.001),
            "smax_stor": float(                 0),
            "affinity_AA_stor": float(          0.001),
            "cmax_stor": float(                 0),
            "affinity_stor_catab": float(       0.001),
            "cmax_AA": float(                 0),
            "affinity_AA_catab": float(       0.001),
            "storage_C_regulation": float(      0.1)
                }

update_N = {
            "r_Nm_AA": float(                   2),
            "r_AA_struct": float(               2),
            "r_AA_stor": float(                 2),
            "xylem_to_root": xylem_to_root,
            "phloem_to_root": phloem_to_root,
            "time_step": int(                   3600),
            }


### Output parameters

plot_N = {
            "p": list([                           'loading_Nm',
                                                  'influx_Nm',
                                                  'Nm'
                        ])
        }

print_g_all = {
            "select": list([                    'influx_Nm',
                                                'loading_Nm',
                                                'soil_Nm',
                                                'Nm',
                                                'z1',
                                                'struct_mass'
                                                # 'C_hexose_root'
                                                # 'thermal_time_since_emergence'
                            ]),
            "vertice": int(                     0)
            }

print_g_one = {
            "select": list([                    'influx_Nm',
                                                'loading_Nm',
                                                'diffusion_Nm_phloem',
                                                'Nm',
                                                'volume',
                                                'z1',
                                                'struct_mass'
                                                # 'C_hexose_root'
                                                # 'thermal_time_since_emergence'
            ]),
            "vertice": int(                     17)
            }
