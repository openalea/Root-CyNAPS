import numpy as np
from rhizodep.nitrogen import ContinuousVessels
from test_mtg import test_mtg, test_nitrogen
from output_display import plot_N, print_g

def init_soil(g,
                    # external conditions parameters
                    zmax_soil_Nm:float = -0.02,
                    soil_Nm_variance:float = 0.0001,
                    soil_Nm_slope:float = 25,
                    scenario:int = 0
                    ):
    props = g.properties()
    props.setdefault('soil_Nm', {})
    soil_Nm = props['soil_Nm']
    z1 = props['z1']

    # No order in update propagation
    max_scale = g.max_scale()
    for vid in g.vertices(scale=max_scale):
        # Soil concentration heterogeneity as border conditions

        soil_Nm[vid] = (
                (0.01 * np.exp(-((z1[vid]-zmax_soil_Nm)**2)/soil_Nm_variance) )**(scenario)
                * (1 + soil_Nm_slope * z1[vid])**(1 - scenario)
                        )
    return g

def test_nitrogen_homogeneous(n=10):
    g = test_mtg()
    g = init_soil(g, scenario=0, soil_Nm_slope=0)

    # Initialization of state variables
    rs = ContinuousVessels(g)

    for i in range(n):
        rs.transport_N()
        rs.update_N()
        # print_g(g, select, vertice=19)

    plot_N(g, p='influx_Nm')
    print_g(g)

    return g


def test_nitrogen_linear(n=10):
    g = test_mtg()
    g = init_soil(g, scenario=0)

    # Initialization of state variables
    rs = ContinuousVessels(g)

    for i in range(n):
        rs.transport_N()
        rs.update_N()
        # print_g(g, select, vertice=19)

    plot_N(g, p='influx_Nm')
    print_g(g)

    return g


def test_nitrogen_patch(n=10):
    g = test_mtg()
    g = init_soil(g, scenario=1)

    # Initialization of state variables
    rs = ContinuousVessels(g)

    for i in range(n):
        rs.transport_N()
        rs.update_N()
        print_g(g, vertice=19)
        # print(g.properties()['Nm'][19])

    plot_N(g, p='influx_Nm')
    print_g(g)

    return g

# Execution
if __name__ == '__main__':
    test_nitrogen_patch()
    # test_nitrogen_linear()
    # test_nitrogen_patch()
    #input('end? ')