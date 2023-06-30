'''IMPORTS'''
from datetime import datetime
import pickle
import xarray as xr
from dataclasses import asdict
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from rhydromin.model_soil import MeanConcentrations, SoilPatch, HydroMinSoil
from rhydromin.model_topology import InitSurfaces, TissueTopology, RadialTopology
from rhydromin.model_water import InitWater, WaterModel
from rhydromin.model_nitrogen import InitCommonN, OnePoolVessels, InitDiscreteVesselsN, DiscreteVessels

from Data_enforcer.model import InitShootNitrogen, InitShootWater, ShootModel

import rhydromin.converter as converter
from rhydromin.tools_output import state_extracts, flow_extracts, global_state_extracts, global_flow_extracts, plot_xr, plot_N
from tools.mtg_dict_to_xarray import mtg_to_dataset, props_metadata


'''FUNCTIONS'''


def N_simulation(init, n, time_step, discrete_vessels=False, plantgl=False, plotting_2D=True, plotting_STM=False, logging=False):
    # Loading mtg file
    with open(init, 'rb') as f:
        g = pickle.load(f)

    # Initialization of modules
    soil = HydroMinSoil(g, **asdict(MeanConcentrations()))
    root_topo = RadialTopology(g, **asdict(InitSurfaces()))
    if not discrete_vessels:
        root_nitrogen = OnePoolVessels(g, **asdict(InitCommonN()))
    else:
        root_water = WaterModel(g, time_step, **asdict(InitWater()))
        root_nitrogen = DiscreteVessels(g, **asdict(InitDiscreteVesselsN()))
    shoot = ShootModel(g, **asdict(InitShootNitrogen()), **asdict(InitShootWater()))

    # Linking modules
    # Spatialized root MTG interactions between soil, structure, nitrogen and water
    converter.link_mtg(root_nitrogen, soil, category="soil", same_names=True)
    converter.link_mtg(root_nitrogen, root_topo, category="structure", same_names=True)

    converter.link_mtg(root_water, soil, category="soil", same_names=True)
    converter.link_mtg(root_water, root_topo, category="structure", same_names=True)

    converter.link_mtg(root_nitrogen, root_water, category="water", same_names=True)

    # 1 point collar interactions between shoot CN, root nitrogen and root water
    converter.link_mtg(shoot, root_nitrogen, category="root_nitrogen", translator=converter.nitrogen_state, same_names=False)
    converter.link_mtg(root_nitrogen, shoot, category="shoot_nitrogen", translator=converter.nitrogen_flows, same_names=False)

    converter.link_mtg(shoot, root_water, category="root_water", translator=converter.water_state, same_names=False)
    converter.link_mtg(root_water, shoot, category="shoot_water", translator=converter.water_flows, same_names=False)

    # Init output xarray list
    if logging:
        # If logging, we start by storing start time and state for later reference during output file analysis
        start_time = datetime.now().strftime("%y.%m.%d_%H.%M")
        xarray_output = [mtg_to_dataset(g, variables=props_metadata, time=0)]
        xarray_output[0].to_netcdf(f"example/outputs/xarray_used_input_{start_time}.nc")

    # Scheduler : actual computation loop
    for i in range(n):
        # Update soil state
        soil.update_patches(patch_age=i*time_step, **asdict(SoilPatch()))
        # Update topological surfaces and volumes based on other evolved structural properties
        root_topo.update_topology(**asdict(TissueTopology()))
        # Compute state variations for water (if selected) and then nitrogen
        if discrete_vessels:
            root_water.exchanges_and_balance()
        root_nitrogen.exchanges_and_balance(time_step=time_step)

        shoot.exchanges_and_balance(time=i)

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
                plot_N(g, list_flow, axs, span_slider=0.1)
            else:
                plot_N(g, list_flow, axs, span_slider=span_slider.val)

        if logging:
            # we build a list of xarray at each time_step as it more efficient than concatenation at each time step
            # However, it might be necessary to empty this and save .nc files every X time steps for memory management
            xarray_output += [mtg_to_dataset(g, variables=props_metadata, time=i+1)]

    if logging:
        # NOTE : merging is slower but way less space is needed
        time_dataset = xr.concat(xarray_output, dim="t")
        time_dataset.to_netcdf(f"example/outputs/{start_time}.nc")

        # saving last mtg status
        with open(r"example/outputs/root{}.pckl".format(str(max(time_dataset.vid.values)).zfill(5)), "wb") as output_file:
            pickle.dump(g, output_file)

        if plotting_2D:
            time_dataset = xr.load_dataset(f"example/outputs/{start_time}.nc")
            plot_xr(dataset=time_dataset, vertice=[1, 3, 5, 7], selection=list(state_extracts.keys()))
            plot_xr(dataset=time_dataset, vertice=[1, 3, 5, 7], selection=list(flow_extracts.keys()))
            plot_xr(dataset=time_dataset, selection=list(global_state_extracts.keys()))
            plot_xr(dataset=time_dataset, selection=list(global_flow_extracts.keys()))
            plt.show()

        if plotting_STM:
            from STM_statistics import STM_analysis
            STM_analysis.run(path=f"example/outputs/{start_time}.nc")

        if plantgl:
            input("end?")
