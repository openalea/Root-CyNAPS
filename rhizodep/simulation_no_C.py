'''IMPORTS'''
from time import sleep
from datetime import datetime
import pickle
import xarray as xr
from dataclasses import asdict
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from rhizodep.model_soil import MeanConcentrations, SoilPatch, HydroMinSoil
from rhizodep.model_topology import InitSurfaces, TissueTopology, RadialTopology
from rhizodep.model_water import InitWater, WaterModel
from rhizodep.model_nitrogen import InitCommonN, OnePoolVessels, InitDiscreteVesselsN, DiscreteVessels

from fakeShoot.model import InitShootNitrogen, InitShootWater, ShootModel

import rhizodep.converter as converter
from rhizodep.tools_output import state_extracts, flow_extracts, plot_xr
from Tools.mtg_dict_to_xarray import output_xarray


'''FUNCTIONS'''
def N_simulation(init, n, time_step, discrete_vessels=False,
                 plotting=True, logging=False):
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
    #print_g(g, **print_g_one)
    #print_g(root_water, ["xylem_total_pressure", "xylem_total_water"], vertice=0)

    if logging:
        start_time = datetime.now().strftime("%y.%m.%d_%H.%M")
        xarray_output = [output_xarray(g, time=0)]
        xarray_output[0].to_netcdf(f"outputs\\xarray_used_input_{start_time}.nc")

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

        print("time step : {}h".format(i))

        if plotting:
            if i == 0:
                plt.ion()
                # legend plot
                fig, axs = plt.subplots(len(plot_properties), 1)
                fig.subplots_adjust(left=0.2, bottom=0.2)

                ax_slider = fig.add_axes([0.25, 0.1, 0.65, 0.03])
                span_slider = Slider(
                    ax=ax_slider,
                    label='Vmax [umol.s-1.m-2]',
                    valmin=0,
                    valmax=1,
                    valinit=0.1)

                [fig.text(0, 0.85 - 0.1 * k, plot_properties[k]) for k in range(len(plot_properties))]
                # actual plot
                plot_N(g, plot_properties, axs, span_slider=0.1)
            else:
                plot_N(g, plot_properties, axs, span_slider=span_slider.val)
            sleep(1e-3)

        if logging:
            # we build a list of xarray at each time_step as it more efficient than concatenation at each time step
            # However, it might be necessary to empty this and save .nc files every X time steps for memory management
            xarray_output += [output_xarray(g, time=i+1)]

        # print_g(g, **print_g_one)
        # print_g(root_water, ["xylem_total_pressure", "xylem_total_water", "water_root_shoot_xylem"], vertice=0)
        # print_g(root_nitrogen, ["Nm_root_shoot_xylem"], vertice=0)

    if plotting:
        input("end? ")

    if logging:
        # NOTE : merging is slower but way less space is needed
        time_dataset = xr.merge(xarray_output)
        #paths = [f"outputs\\{t}.nc" for t in range(len(xarray_output))]
        #xr.save_mfdataset(xarray_output, paths=paths)
        time_dataset.to_netcdf(f"outputs\\{start_time}.nc")

        # saving last mtg status
        with open(r"outputs\\root{}.pckl".format(str(max(time_dataset.vid.values)).zfill(5)), "wb") as output_file:
            pickle.dump(g, output_file)

        time_dataset = xr.load_dataset(f"outputs\\{start_time}.nc")
        print(time_dataset)
        plot_xr(dataset=time_dataset, vertice=149, select_state=state_extracts)

