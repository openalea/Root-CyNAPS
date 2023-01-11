import numpy as np
from time import sleep
from rhizodep.nitrogen import OnePoolVessels
import rhizodep.parameters_nitrogen as Nparam
from test_mtg import test_mtg
from output_display import plot_N, print_g


def init_soil_N(g, zmax_soil_Nm, soil_Nm_variance, soil_Nm_slope, scenario):

    props = g.properties()
    props.setdefault('soil_Nm', {})
    props.setdefault('soil_AA', {})
    soil_Nm = props['soil_Nm']
    soil_AA = props['soil_AA']
    z1 = props['z1']

    # No order in update propagation
    max_scale = g.max_scale()
    for vid in g.vertices(scale=max_scale):
        # Soil concentration heterogeneity as border conditions

        soil_Nm[vid] = (
                (0.01 * np.exp(-((z1[vid] - zmax_soil_Nm) ** 2) / soil_Nm_variance)) ** (scenario)
                * (1 + soil_Nm_slope * z1[vid]) ** (1 - scenario)
        )

        soil_AA[vid] = 1e-3

    return g


def test_nitrogen_scenario(n, scenario):
    g = test_mtg()
    g = init_soil_N(g, **scenario)

    # Initialization of state variables
    rs = OnePoolVessels(g, **Nparam.init_common_N)

    for i in range(n):
        rs.transport_N(**Nparam.transport_common_N)
        # N metabolism is not yet computed as C is not actualized yet.
        # rs.metabolism_N(**Nparam.metabolism_N)
        rs.update_N(**Nparam.update_N)

        if i == 0:
            rng_min, rng_max = [0 for k in Nparam.plot_N['p']], [0 for k in Nparam.plot_N['p']]
            rng_min, rng_max = plot_N(g, rng_min, rng_max, **Nparam.plot_N)
        else:
            plot_N(g, rng_min, rng_max, **Nparam.plot_N)
        print_g(g, **Nparam.print_g_one)
        sleep(0.1)



    #plot_N(g, **Nparam.plot_N)
    #print_g(g, **Nparam.print_g_all)

    return g


# Execution
if __name__ == '__main__':
    test_nitrogen_scenario(n=30, scenario=Nparam.init_soil_patch)
    input('end? ')
