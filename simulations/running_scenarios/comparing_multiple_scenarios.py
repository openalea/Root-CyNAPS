import os
import pickle

import xarray as xr

from root_cynaps.tools import plot_mtg
import openalea.plantgl.all as pgl
from openalea.mtg.traversal import pre_order, post_order

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

    # Plotting topological plan
    # TODO, recenter and colorbar legend ?
    print("[INFO] Plotting topology and coordinate map...")
    # Loading MTG file
    g_name = [name for name in os.listdir(working_dir) if ".pckl" in name][0]
    with open(working_dir + '/' + g_name, 'rb') as f:
        g = pickle.load(f)
    # Plotting the vid as color to enable a better reading of groups
    g.properties()["v"] = dict(
        zip(list(g.properties()["struct_mass"].keys()), list(g.properties()["struct_mass"].keys())))
    scene = pgl.Scene()
    scene += plot_mtg(g,
                      prop_cmap="v",
                      lognorm=False,  # to avoid issues with negative values
                      vmin=min(g.properties()["struct_mass"].keys()),
                      vmax=max(g.properties()["struct_mass"].keys()),
                      k=0)
    pgl.Viewer.display(scene)

    # # TESTS FOR PERSISTENT HOMOLOGY COORDINATES
    # root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    # root = next(root_gen)
    # ct = 0
    # g.properties()["dist_to_collar"] = {1: 0}
    # for vid in pre_order(g, root):
    #     if vid != 1:
    #         parent = g.parent(vid)
    #         g.properties()["dist_to_collar"][vid] = g.properties()["dist_to_collar"][parent] + g.properties()["length"][vid]
    #
    # print(g.properties()["dist_to_collar"])
    #
    # def whole_axis(g, vid, prop):
    #     axes_list = []
    #     axis_dict = {}
    #     axis_dict[vid] = g.properties()[prop][vid]
    #     while g.properties()["label"][vid] != "Apex":
    #         for v in g.children(vid):
    #             if g.properties()["edge_type"][v] != "+":
    #                 vid = v
    #             else:
    #                 axes_list += whole_axis(g, v, prop)
    #         axis_dict[vid] = g.properties()[prop][vid]
    #     axes_list += [axis_dict]
    #     return axes_list
    #
    # axes_list = whole_axis(g=g, vid=2, prop="dist_to_collar")
    # axes_insertion = [g.parent(list(axe.keys())[0]) for axe in axes_list]
    #
    # def ordinator(axe):
    #     axe_po_groups = [axe]
    #     for v in list(axe.keys()):
    #         if v in axes_insertion:
    #             axe_po_groups += ordinator(axe=axes_list[axes_insertion.index(v)])
    #     return axe_po_groups
    #
    # ordered_axes_list = ordinator(axe = axes_list[-1])
    #
    # fig, ax = plt.subplots()
    # for axe in range(len(ordered_axes_list)):
    #     X = list(ordered_axes_list[axe].values())
    #     Y = [axe for k in range(len(X))]
    #     ax.plot(X, Y, '.-', linewidth=2, markersize=8)
    # fig.show()

    # TODO : general sensitivity analysis on time-series data
    # Global sensitivity analysis at the end of the simulation for now
    # Using a linear regression

    print("[INFO] Performing regression sensitivity on model final global states...")
    from tools import global_sensitivity
    global_sensitivity.regression_analysis(dataset=central_dataset, output_path=working_dir)

    # TODO : Plotting global outputs, should work with minor adjustments
    print("Plotting global properties...")
    # plot_xr(datasets=datasets, selection=list(global_state_extracts.keys()), supplementary_legend=supplementary_legend)
    # plot_xr(datasets=datasets, selection=list(global_flow_extracts.keys()), supplementary_legend=supplementary_legend)

    # Running STM sensitivity analysis
    # For some reason, dataset should be loaded before umap, and the run() call should be made at the end of
    # the workflow because tkinter locks everything
    # TODO : adapt to sliding windows along roots ?
    print("[INFO] Performing local organs' physiology clustering...")
    from tools import STM_analysis
    STM_analysis.run(file=central_dataset, output_path=working_dir)


if __name__ == '__main__':
    analyze_multiple_scenarios(scenarios_set=0)
