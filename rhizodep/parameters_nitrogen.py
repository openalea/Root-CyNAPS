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
            "Nm": float(                        1e-5),
            "influx_Nm": float(                 0),
            "loading_Nm": float(                0),
            "xylem_Nm": float(                  0.1),
            "xylem_volume": float(              5e-10)
          }


### Model parameters

# Global parameters
xylem_to_root: float = 0.2

transport_N = {
            # kinetic parameters
            "affinity_Nm_root": float(          2),
            "vmax_Nm_emergence": float(         0.1),
            "affinity_Nm_xylem": float(         0.1),
            # metabolism-related parameters
            "transport_C_regulation": float(    1e-2),
            "transport_N_regulation": float(    0.01),
            # architecture parameters
            "xylem_to_root": xylem_to_root,
            "epiderm_differentiation": float(   1e-6),
            "endoderm_differentiation": float(  1e-6)
                }

update_N = {
            "xylem_to_root": xylem_to_root,
            "time_step": int(                   3600),
            }


### Output parameters

plot_N = {
            "p": str(                           "Nm")
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
                                                'soil_Nm',
                                                'Nm',
                                                'z1',
                                                'struct_mass'
                                                # 'C_hexose_root'
                                                # 'thermal_time_since_emergence'
            ]),
            "vertice": int(                     19)
            }
