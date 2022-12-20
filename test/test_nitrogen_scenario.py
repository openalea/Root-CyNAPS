import numpy as np
from rhizodep.nitrogen import init_N, transport_N, update_N
from test_mtg import test_mtg, init_N, test_nitrogen
from output_display import plot_N, print_g

def init_soil(g,
                    # external conditions parameters
                    zmax_soil_Nm:float = -0.02,
                    soil_Nm_variance:float = 0.0001,
                    soil_Nm_slope:float = 25,
                    scenario:int = 0
                    ):
    props = g.properties()
    soil_Nm = props['soil_Nm']
    z1 = props['z1']

    # No order in update propagation
    max_scale = g.max_scale()
    for vid in g.vertices(scale=max_scale):
        # Soil concentration heterogeneity as border conditions

        soil_Nm[vid] = (
                ( 0.01 * np.exp(-((z1[vid]-zmax_soil_Nm)**2)/soil_Nm_variance) )**(scenario)
                * (1 + soil_Nm_slope * z1[vid])**(1 - scenario)
                        )

    return g

def test_nitrogen_homogeneous(n=10):
    g = test_mtg()

    # Initialization of state variable
    g = init_N(g)
    g = init_soil(g, scenario=0, soil_Nm_slope=0)

    for i in range(n):
        g = transport_N(g)
        g = update_N(g)
        # print_g(g, select, vertice=19)
    plot_N(g, p='influx_Nm')

    print_g(g)
    print(g.node(0).xylem_Nm, g.node(0).xylem_volume)

    return g


def test_nitrogen_linear(n=10):
    g = test_mtg()

    # Initialization of state variable
    g = init_N(g)
    g = init_soil(g, scenario=0)

    for i in range(n):
        g = transport_N(g)
        g = update_N(g)
        # print_g(g, select, vertice=19)
    plot_N(g, p='influx_Nm')

    print_g(g)
    print(g.node(0).xylem_Nm, g.node(0).xylem_volume)

    return g


def test_nitrogen_patch(n=10):
    g = test_mtg()

    # Initialization of state variable
    g = init_N(g)
    g = init_soil(g, scenario=1)

    for i in range(n):
        g = transport_N(g)
        g = update_N(g)
        # print_g(g, select, vertice=19)
    plot_N(g, p='influx_Nm')

    print_g(g)
    print(g.node(0).xylem_Nm, g.node(0).xylem_volume)

    return g

# Execution
if __name__ == '__main__':
    test_nitrogen_homogeneous()
    # test_nitrogen_linear()
    # test_nitrogen_patch()
    #input('end? ')