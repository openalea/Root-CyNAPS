# Imports
from time import sleep
import pickle
from dataclasses import asdict

from rhizodep.model_soil import MeanConcentrations, SoilPatch, SoilNitrogen
from rhizodep.model_topology import InitSurfaces, TissueTopology, RadialTopology
from rhizodep.model_water import InitWater, WaterModel
from rhizodep.model_nitrogen import InitCommonN, OnePoolVessels


from rhizodep.tools_output import plot_properties, print_g_one, plot_N, print_g


def N_simulation(init, n, time_step, outside_flows,
                 outputs=True):
    with open(init, 'rb') as f:
        g = pickle.load(f)
    output = input("plot results? (y/n) :")
    # Initialization of state variables
    soil = SoilNitrogen(g, **asdict(MeanConcentrations()))
    root_topo = RadialTopology(g, **asdict(InitSurfaces()))
    #root_water = WaterModel(g, time_step, **asdict(InitWater()))
    root_nitrogen = OnePoolVessels(g, **asdict(InitCommonN()), **outside_flows)

    for i in range(n):
        soil.update_patches(patch_age=i*time_step, **asdict(SoilPatch()))
        root_topo.update_topology(**asdict(TissueTopology()))
        #root_water.exchanges_and_balance()
        root_nitrogen.exchanges_and_balance(time_step=time_step)

        if output == 'y':
            if i == 0:
                rng_min, rng_max = [0 for k in plot_properties], [0 for k in plot_properties]
                rng_min, rng_max = plot_N(g, rng_min, rng_max, plot_properties)
            else:
                plot_N(g, rng_min, rng_max, plot_properties)
        print_g(g, **print_g_one)
        sleep(0.01)

    #print_g(g, **print_g_all)
    input("end? ")
