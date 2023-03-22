# Imports
from time import sleep
import pickle
from dataclasses import asdict

from rhizodep.model_soil import MeanConcentrations, SoilPatch, HydroMinSoil
from rhizodep.model_topology import InitSurfaces, TissueTopology, RadialTopology
from rhizodep.model_water import InitWater, WaterModel
from rhizodep.model_nitrogen import InitCommonN, OnePoolVessels, InitDiscreteVesselsN, DiscreteVessels

from fakeShoot.model import InitShootNitrogen, InitShootWater, ShootModel

import rhizodep.converter as converter
from rhizodep.tools_output import plot_properties, print_g_one, plot_N, print_g


def N_simulation(init, n, time_step, discrete_vessels=False,
                 plotting=True):
    with open(init, 'rb') as f:
        g = pickle.load(f)
    # Initialization of state variables
    soil = HydroMinSoil(g, **asdict(MeanConcentrations()))
    root_topo = RadialTopology(g, **asdict(InitSurfaces()))
    if not discrete_vessels:
        root_nitrogen = OnePoolVessels(g, **asdict(InitCommonN()), **asdict(InitShootNitrogen()))
    else:
        root_water = WaterModel(g, time_step, **asdict(InitWater()), **asdict(InitShootWater()))
        root_nitrogen = DiscreteVessels(g, **asdict(InitDiscreteVesselsN()), **asdict(InitShootNitrogen()))
    shoot = ShootModel(**asdict(InitShootNitrogen()), **asdict(InitShootWater()))

    for i in range(n):
        soil.update_patches(patch_age=i*time_step, **asdict(SoilPatch()))
        root_topo.update_topology(**asdict(TissueTopology()))
        if discrete_vessels:
            root_water.exchanges_and_balance()
        root_nitrogen.exchanges_and_balance(time_step=time_step)

        # to be retrieved by the shoot model
        nitrogen_state = converter.get_root_collar_state(root_nitrogen)
        if discrete_vessels:
            water_state = converter.get_root_collar_state(root_water)
            collar_nitrogen_flows, collar_water_flows = shoot.exchanges_and_balance(**{**nitrogen_state, **water_state})
            converter.apply_root_collar_flows(collar_water_flows, root_water, "water")
        else:
            #WRONG!!
            collar_nitrogen_flows, collar_water_flows = shoot.exchanges_and_balance(root_xylem_pressure=0, **nitrogen_state)
        converter.apply_root_collar_flows(collar_nitrogen_flows, root_nitrogen, "nitrogen")

        if plotting:
            if i == 0:
                rng_min, rng_max = [0 for k in plot_properties], [0 for k in plot_properties]
                rng_min, rng_max = plot_N(g, rng_min, rng_max, plot_properties)
            else:
                plot_N(g, rng_min, rng_max, plot_properties)
        print_g(g, **print_g_one)
        sleep(0.01)

    #print_g(g, **print_g_all)
    input("end? ")
