from time import sleep
from rhizodep.model_soil import MeanConcentrations, SoilPatch, SoilNitrogen
from rhizodep.model_nitrogen import InitCommonN, OnePoolVessels
from rhizodep.model_topology import InitSurfaces, TissueTopology, RadialTopology
from test.test_mtg import test_mtg
from tools_output import plot_properties, print_g_one, plot_N, print_g
from dataclasses import asdict


def N_simulation(n, time_step):
    g = test_mtg()
    output = input("plot results? (y/n) :")

    # Initialization of state variables
    soil_nitrogen = SoilNitrogen(g, **asdict(MeanConcentrations()))
    root_topo = RadialTopology(g, **asdict(InitSurfaces()))
    root_nitrogen = OnePoolVessels(g, **asdict(InitCommonN()))

    for i in range(n):
        soil_nitrogen.update_patches(patch_age=i*time_step, **asdict(SoilPatch()))
        root_topo.update_topology(**asdict(TissueTopology()))
        root_nitrogen.exchanges_and_balance()

        if output == 'y':
            if i == 0:
                rng_min, rng_max = [0 for k in plot_properties], [0 for k in plot_properties]
                rng_min, rng_max = plot_N(g, rng_min, rng_max, plot_properties)
            else:
                plot_N(g, rng_min, rng_max, plot_properties)
        print_g(g, **print_g_one)
        sleep(0.1)

    #print_g(g, **print_g_all)

    return g


# Execution
if __name__ == '__main__':
    N_simulation(n=20, time_step=3600)
    input('end? ')
