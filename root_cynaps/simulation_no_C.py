'''IMPORTS'''
import pickle
import os
import shutil
import xarray as xr
from dataclasses import asdict
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from root_cynaps.root_cynaps.model_soil import MeanConcentrations, SoilPatch, HydroMinSoil
from root_cynaps.root_cynaps.model_topology import InitSurfaces, TissueTopology, RadialTopology
from root_cynaps.root_cynaps.model_water import InitWater, WaterModel
from root_cynaps.root_cynaps.model_nitrogen import InitDiscreteVesselsN, DiscreteVessels

from root_cynaps.Data_enforcer.model import ShootModel

import root_cynaps.root_cynaps.converter as converter
from root_cynaps.root_cynaps.tools_output import state_extracts, flow_extracts, global_state_extracts, global_flow_extracts, plot_xr, plot_N
from root_cynaps.tools.mtg_dict_to_xarray import mtg_to_dataset, props_metadata


'''FUNCTIONS'''


def N_simulation(hexose_decrease_rate, z_soil_Nm_max, output_path, current_file_dir, init, steps_number, time_step, echo=False,
                 plantgl=False, plotting_2D=True, plotting_STM=False, logging=False, max_time_steps_for_memory=100):
    # Store this before anything else to ensure the locals order is right
    Loc = locals()
    real_parameters = ["output_path", "current_file_dir", "init", "n", "time_step", "echo", "plantgl", "plotting_2D", "plotting_STM", "logging", "max_time_steps_for_memory"]
    scenario = dict([(key, value) for key, value in Loc.items() if key not in real_parameters])

    # Loading mtg file
    with open(current_file_dir + "/inputs/" + init, 'rb') as f:
        g = pickle.load(f)

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
        time_xrs = [mtg_to_dataset(g, variables=props_metadata, time=0)]
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
        root_nitrogen.exchanges_and_balance(hexose_decrease_rate)

        shoot.exchanges_and_balance(time=i)

        if echo:
            print("time step : {}h".format(i))

        if plantgl:
            if i == 0:
                plt.ion()
                # legend plot
                fig, axs = plt.subplots(len(flow_extracts), 1)
                fig.subplots_adjust(left=0.2, bottom=0.2)

                ax_slider = fig.add_axes([0.25, 0.1, 0.65, 0.03])
                span_slider = Slider(
                    ax=ax_slider,
                    label='Vmax [umol.s-1.m-2]',
                    valmin=0,
                    valmax=1,
                    valinit=0.1)

                list_flow = list(flow_extracts.keys())
                [fig.text(0, 0.85 - 0.1 * k, list_flow[k]) for k in range(len(list_flow))]
                # actual plot
                plot_N(g, list_flow, axs)
            else:
                plot_N(g, list_flow, axs, span_slider=span_slider.val)

        if logging:
            # we build a list of xarray at each time_step as it more efficient than concatenation at each time step
            # However, it might be necessary to empty this and save .nc files every X time steps for memory management
            time_xrs += [mtg_to_dataset(g, variables=props_metadata, time=i+1)]
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
        # NOTE : merging is slower but way less space is needed
        time_step_files = [output_path[:-3] + '/' + name for name in os.listdir(output_path[:-3])]
        time_dataset = xr.open_mfdataset(time_step_files)
        time_dataset = time_dataset.assign_coords(coords=scenario).expand_dims(dim=dict(zip(list(scenario.keys()), [1 for k in scenario])))
        time_dataset.to_netcdf(output_path)

        if plotting_2D:
            time_dataset = xr.load_dataset(output_path)
            #plot_xr(datasets=time_dataset, vertice=[1, 3, 5, 7, 9], selection=list(state_extracts.keys()))
            #plot_xr(datasets=time_dataset, vertice=[1, 3, 5, 7, 9], selection=list(flow_extracts.keys()))
            plot_xr(datasets=time_dataset, selection=list(global_state_extracts.keys()))
            plot_xr(datasets=time_dataset, selection=list(global_flow_extracts.keys()))
            plt.show()

        if plotting_STM:
            from tools import STM_analysis
            STM_analysis.run(path=output_path)

        del time_dataset
        shutil.rmtree(output_path[:-3])

        if plantgl:
            input("end?")
