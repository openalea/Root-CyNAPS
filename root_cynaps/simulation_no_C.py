'''IMPORTS'''
import pickle
import os
import shutil
import xarray as xr
from dataclasses import asdict
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import inspect

from root_cynaps.model_soil import MeanConcentrations, SoilPatch, HydroMinSoil
from root_cynaps.model_topology import InitSurfaces, TissueTopology, RadialTopology
from root_cynaps.model_water import InitWater, WaterModel
from root_cynaps.model_nitrogen import InitDiscreteVesselsN, DiscreteVessels

from Data_enforcer.model import ShootModel

import root_cynaps.converter as converter
from root_cynaps.output_properties import state_extracts, flow_extracts, global_state_extracts, global_flow_extracts
from tools.mtg_dict_to_xarray import mtg_to_dataset

from statistical_tools.main import launch_analysis

'''FUNCTIONS'''


def N_simulation(z_soil_Nm_max, output_path, current_file_dir, init, steps_number, time_step, echo=False,
                 plotting_2D=True, plotting_STM=False, logging=False, max_time_steps_for_memory=100):
    # Store this before anything else to ensure the locals order is right
    Loc = locals()
    real_parameters = ["output_path", "current_file_dir", "init", "steps_number", "time_step", "echo", "plotting_2D", "plotting_STM", "logging", "max_time_steps_for_memory"]
    scenario = dict([(key, value) for key, value in Loc.items() if key not in real_parameters])

    # Loading mtg file
    with open(current_file_dir + "/inputs/" + init, 'rb') as f:
        g = pickle.load(f)

    #g._properties = mtg_to_dataset(g, variables=dict(struct_mass=dict(unit="g", value_example=0.000134696, description="not provided")), time=0)
    #test = g.properties()

    #vm = 10
    #km = 1
    #def f(i, v):
    #    return i.struct_mass * v

    #test["struct_mass"].loc[dict(vid=1, t=0)] = 0
    #test.update(test.assign(dict(struct_m2=lambda x: f(x, vm), struct_m3=lambda x: f(x, vm))))

    #with open(current_file_dir + "/outputs/test.pckl", "wb") as f:
    #    pickle.dump(g, f)

    # with open(current_file_dir + "/outputs/test.pckl", 'rb') as f:
    #     g = pickle.load(f)

    # print(g.properties())

    # Output variables for logs
    log_outputs = {}
    for d in [state_extracts, flow_extracts, global_state_extracts, global_flow_extracts]:
        log_outputs.update(d)

    # Initialization of modules
    soil = HydroMinSoil(g, **asdict(MeanConcentrations()))
    root_topo = RadialTopology(g, **asdict(InitSurfaces()))
    root_water = WaterModel(g, time_step, **asdict(InitWater()))
    root_nitrogen = DiscreteVessels(g, time_step, **asdict(InitDiscreteVesselsN()))
    shoot = ShootModel(g)

    # Linking modules
    # Spatialized root MTG interactions between soil, structure, nitrogen and water
    converter.link_mtg(root_nitrogen, soil, category="soil", same_names=True)
    converter.link_mtg(root_nitrogen, root_topo, category="structure", same_names=True)

    converter.link_mtg(root_water, soil, category="soil", same_names=True)

    converter.link_mtg(root_water, root_topo, category="structure", same_names=True)

    converter.link_mtg(root_nitrogen, root_water, category="water", same_names=True)

    # 1 point collar interactions between shoot CN, root nitrogen and root water
    converter.link_mtg(root_nitrogen, shoot, category="shoot_nitrogen", translator=converter.nitrogen_flows, same_names=False)

    converter.link_mtg(root_water, shoot, category="shoot_water", translator=converter.water_flows, same_names=False)

    # Init output xarray list
    if logging:
        os.mkdir(output_path[:-3])
        time_xrs = [mtg_to_dataset(g, variables=log_outputs, time=0)]
        # xarray_output[0].to_netcdf(output_path + f"/xarray_used_input_{start_time}.nc")

    root_water.init_xylem_water()
    # Scheduler : actual computation loop
    for i in range(steps_number):
        # Update soil state
        soil.update_patches(patch_age=i*time_step, z_soil_Nm_max=z_soil_Nm_max, **asdict(SoilPatch()))
        # Update topological surfaces and volumes based on other evolved structural properties
        root_topo.update_topology(**asdict(TissueTopology()))
        # Compute state variations for water (if selected) and then nitrogen
        root_water.exchanges_and_balance()
        root_nitrogen.exchanges_and_balance()

        shoot.exchanges_and_balance(time=i)

        if echo:
            print("time step : {}h".format(i))

        if logging:
            # we build a list of xarray at each time_step as it more efficient than concatenation at each time step
            time_xrs += [mtg_to_dataset(g, variables=log_outputs, time=i+1)]
            if len(time_xrs) >= max_time_steps_for_memory:
                interstitial_dataset = xr.concat(time_xrs, dim="t")
                interstitial_dataset.to_netcdf(output_path[:-3] + f'/t={i+1}.nc')
                del interstitial_dataset
                del time_xrs
                time_xrs = []

    if logging:
        if len(time_xrs) > 0:
            interstitial_dataset = xr.concat(time_xrs, dim="t")
            interstitial_dataset.to_netcdf(output_path[:-3] + f'/tf.nc')
            del interstitial_dataset
            del time_xrs

        # SAVING and merging
        # NOTE : merging is slower but way less space is needed
        time_step_files = [output_path[:-3] + '/' + name for name in os.listdir(output_path[:-3])]
        time_dataset = xr.open_mfdataset(time_step_files)
        time_dataset = time_dataset.assign_coords(coords=scenario).expand_dims(dim=dict(zip(list(scenario.keys()), [1 for k in scenario])))
        time_dataset.to_netcdf(output_path[:-3] + '/merged.nc')
        del time_dataset
        for file in os.listdir(output_path[:-3]):
            if '.nc' in file and file != "merged.nc":
                os.remove(output_path[:-3] + '/' + file)

        time_dataset = xr.load_dataset(output_path[:-3] + '/merged.nc')
        # Launching outputs analyses
        launch_analysis(dataset=time_dataset, mtg=g, output_dir=output_path[:-3],
                        global_state_extracts=global_state_extracts, global_flow_extracts=global_flow_extracts,
                        state_extracts=state_extracts, flow_extracts=flow_extracts,
                        global_sensitivity=False, global_plots=plotting_2D, STM_clustering=plotting_STM)
