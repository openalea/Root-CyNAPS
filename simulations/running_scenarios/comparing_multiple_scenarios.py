import os
import pickle

import xarray as xr

from root_cynaps.tools import plot_mtg
from root_cynaps.tools_output import plot_xr, global_state_extracts, global_flow_extracts
import openalea.plantgl.all as pgl

import matplotlib.pyplot as plt


def analyze_multiple_scenarios(scenarios_set):
    # Setting the working dir to current file' outputs subdirectory
    root_path = os.path.dirname(__file__)
    possible_scenarios = os.listdir(root_path + '/outputs')

    # Reading scenario instructions
    if scenarios_set == -1:
        for k in range(len(possible_scenarios)):
            print(k, possible_scenarios[k])
        scenarios_set = int(input("which scenarios ?"))
    working_dir = root_path + '/outputs/' + possible_scenarios[scenarios_set]

    central_dataset = xr.load_dataset(working_dir + '/merged.nc')

    # TERMINAL SENSITIVITY ANALYSIS
    # TODO : general sensitivity analysis on time-series data, but issue of post simulation Sensitivity Methods not existing
    # Global sensitivity analysis at the end of the simulation for now
    # Using a linear regression

    print("[INFO] Performing regression sensitivity on model final global states...")
    from tools import global_sensitivity
    global_sensitivity.regression_analysis(dataset=central_dataset, output_path=working_dir)

    # PLOTTING GLOBAL OUTPUTS
    print("[INFO] Plotting global properties...")
    plot_xr(datasets=central_dataset, selection=list(global_state_extracts.keys()))
    plot_xr(datasets=central_dataset, selection=list(global_flow_extracts.keys()))

    # PLOTTING ARCHITECTURED VID LEGEND
    print("[INFO] Plotting topology and coordinate map...")
    # Loading MTG file
    g_name = [name for name in os.listdir(working_dir) if ".pckl" in name][0]
    with open(working_dir + '/' + g_name, 'rb') as f:
        g = pickle.load(f)
    # Plotting the vid as color to enable a better reading of groups
    g.properties()["v"] = dict(
        zip(list(g.properties()["struct_mass"].keys()), list(g.properties()["struct_mass"].keys())))

    from tools.custom_colorbar import custom_colorbar
    custom_colorbar(min(g.properties()["v"].values()), max(g.properties()["v"].values()), unit="Vid number")

    scene = pgl.Scene()
    scene += plot_mtg(g,
                      prop_cmap="v",
                      lognorm=False,  # to avoid issues with negative values
                      vmin=min(g.properties()["struct_mass"].keys()),
                      vmax=max(g.properties()["struct_mass"].keys()))
    pgl.Viewer.display(scene)
    pgl.Viewer.saveSnapshot(working_dir + "/vid_map.png")

    # RUNNING STM CLUSTERING AND SENSITIVITY ANALYSIS
    # For some reason, dataset should be loaded before umap, and the run() call should be made at the end of
    # the workflow because tkinter locks everything
    # TODO : adapt to sliding windows along roots ?
    print("[INFO] Performing local organs' physiology clustering...")
    from tools import STM_analysis
    STM_analysis.run(file=central_dataset, output_path=working_dir)


if __name__ == '__main__':
    analyze_multiple_scenarios(scenarios_set=0)
