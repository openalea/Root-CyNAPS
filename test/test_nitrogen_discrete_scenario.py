import numpy as np
from time import sleep
from rhizodep.nitrogen import ContinuousVessels
import rhizodep.parameters_nitrogen as Nparam
from test_mtg import test_mtg, test_nitrogen
from output_display import plot_N, print_g


def init_soil(g, zmax_soil_Nm, soil_Nm_variance, soil_Nm_slope, scenario):

    props = g.properties()
    props.setdefault('soil_Nm', {})
    props.setdefault('W_potential', {})
    soil_Nm = props['soil_Nm']
    W_potential = props['W_potential']
    z1 = props['z1']

    # No order in update propagation
    max_scale = g.max_scale()
    for vid in g.vertices(scale=max_scale):
        # Soil concentration heterogeneity as border conditions

        soil_Nm[vid] = (
                (0.01 * np.exp(-((z1[vid] - zmax_soil_Nm) ** 2) / soil_Nm_variance)) ** (scenario)
                * (1 + soil_Nm_slope * z1[vid]) ** (1 - scenario)
        )

        W_potential[vid] = (
                (0.01 * np.exp(-((z1[vid] - zmax_soil_Nm) ** 2) / soil_Nm_variance)) ** (scenario)
                * (1 + soil_Nm_slope * z1[vid]) ** (1 - scenario)
        )
    return g


def test_nitrogen_scenario(n, scenario):
    g = test_mtg()
    g = init_soil(g, **scenario)

    # Initialization of state variables
    rs = ContinuousVessels(g, **Nparam.init_N)

    for i in range(n):
        rs.transport_N(**Nparam.transport_N)
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
    test_nitrogen_scenario(n=20, scenario=Nparam.init_soil_patch)
    input('end? ')
