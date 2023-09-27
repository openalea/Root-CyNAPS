import os

import matplotlib.pyplot as plt
import xarray as xr

from root_cynaps.tools_output import plot_xr, global_state_extracts, global_flow_extracts


def plot_multiple_scenario(scenarios_set):
    # READING SCENARIO INSTRUCTIONS:
    root_path = os.path.dirname(__file__)
    working_dir = os.listdir(root_path + '/outputs')
    if scenarios_set == -1:
        for k in range(len(working_dir)):
            print(k, working_dir[k])
        scenarios_set = int(input("which scenarios ?"))
    list_dir = os.listdir(root_path + '/outputs/' + working_dir[scenarios_set])
    time_datasets = []
    legend = []
    for directory in list_dir:
        for filename in os.listdir(root_path + '/outputs/' + working_dir[scenarios_set] + '/' + directory):
            if ".nc" in filename and "xarray_used_input" not in filename:
                filepath = root_path + '/outputs/' + working_dir[scenarios_set] + '/' + directory + '/' + filename
                time_datasets += [xr.load_dataset(filepath)]
        legend += [directory]

    # plot global properties
    plot_xr(datasets=time_datasets, selection=list(global_state_extracts.keys()), supplementary_legend=legend)
    plot_xr(datasets=time_datasets, selection=list(global_flow_extracts.keys()), supplementary_legend=legend)
    # plot local properties
    plt.show()


def plantgl_multiple_scenario(time_steps=[]):
    return


if __name__ == '__main__':
    plot_multiple_scenario(scenarios_set=0)
