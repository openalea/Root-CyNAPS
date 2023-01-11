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
