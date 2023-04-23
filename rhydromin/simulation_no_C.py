'''IMPORTS'''
from time import sleep
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

from fakeShoot.model import InitShootNitrogen, InitShootWater, ShootModel

import rhydromin.converter as converter
from rhydromin.tools_output import state_extracts, flow_extracts, plot_xr, plot_N
from tools.mtg_dict_to_xarray import mtg_to_dataset, globals_to_dataset


'''FUNCTIONS'''


def N_simulation(init, n, time_step, discrete_vessels=False, plantgl=False, plotting=True, logging=False):
    # Loading mtg file
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

    # To visualize proper initialization
    #print_g(root_water, ["xylem_total_pressure", "xylem_total_water"], vertice=0)

    if logging:
        # If logging, we start by storing start time and state for later reference during output file analysis
        start_time = datetime.now().strftime("%y.%m.%d_%H.%M")
        xarray_output = [mtg_to_dataset(g, time=0)]
        xarray_glob_output = [globals_to_dataset(root_water, time=0)]
        xarray_output[0].to_netcdf(f"outputs\\xarray_used_input_{start_time}.nc")

    # actual computation loop
    for i in range(n):
        # Update soil state
        soil.update_patches(patch_age=i*time_step, **asdict(SoilPatch()))
        # Update topological surfaces and volumes based on other evolved structural properties
        root_topo.update_topology(**asdict(TissueTopology()))
        # Compute state variations for water (if selected) and then nitrogen
        if discrete_vessels:
            root_water.exchanges_and_balance()
        root_nitrogen.exchanges_and_balance(time_step=time_step)

        # to be retrieved by the shoot model
        nitrogen_state = converter.get_root_collar_state(root_nitrogen)
        if discrete_vessels:
            water_state = converter.get_root_collar_state(root_water)
            collar_nitrogen_flows, collar_water_flows = shoot.exchanges_and_balance(**{**nitrogen_state, **water_state})
            # apply computed water flow for next time step to root model
            converter.apply_root_collar_flows(collar_water_flows, root_water, "water")
        else:
            # Here only nitrogen flows are retrieved
            collar_nitrogen_flows = shoot.exchanges_and_balance(root_xylem_pressure=0, **nitrogen_state)[0]
        # apply computed nitrogen flow for next time step to root model
        converter.apply_root_collar_flows(collar_nitrogen_flows, root_nitrogen, "nitrogen")

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

                [fig.text(0, 0.85 - 0.1 * k, flow_extracts[k]) for k in range(len(flow_extracts))]
                # actual plot
                plot_N(g, flow_extracts, axs, span_slider=0.1)
            else:
                plot_N(g, flow_extracts, axs, span_slider=span_slider.val)

        if logging:
            # we build a list of xarray at each time_step as it more efficient than concatenation at each time step
            # However, it might be necessary to empty this and save .nc files every X time steps for memory management
            xarray_output += [mtg_to_dataset(g, time=i+1)]
            xarray_glob_output += [globals_to_dataset(root_water, time=i+1)]

        # print_g(root_water, ["xylem_total_pressure", "xylem_total_water", "water_root_shoot_xylem"], vertice=0)
        # print_g(root_nitrogen, ["Nm_root_shoot_xylem"], vertice=0)

    if logging:
        # NOTE : merging is slower but way less space is needed
        time_dataset = xr.concat(xarray_output, dim="t")
        time_glob_dataset = xr.concat(xarray_glob_output, dim="t")
        time_dataset.to_netcdf(f"outputs\\{start_time}.nc")
        time_glob_dataset.to_netcdf(f"outputs\\{start_time}_glob.nc")

        # saving last mtg status
        with open(r"outputs\\root{}.pckl".format(str(max(time_dataset.vid.values)).zfill(5)), "wb") as output_file:
            pickle.dump(g, output_file)

        if plotting:
            time_dataset = xr.load_dataset(f"outputs\\{start_time}.nc")
            time_glob_dataset = xr.load_dataset(f"outputs\\{start_time}_glob.nc")
            plot_xr(dataset=time_dataset, vertice=[1, 49, 149, 249], selection=list(state_extracts.keys()))
            plot_xr(dataset=time_dataset, vertice=[1, 49, 149, 249], selection=list(flow_extracts.keys()))
            plt.show()
