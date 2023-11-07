import os
import pickle

import xarray as xr

from statistical_tools.main import launch_analysis


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

    # Loading MTG file
    g_name = [name for name in os.listdir(working_dir) if ".pckl" in name][0]
    with open(working_dir + '/' + g_name, 'rb') as f:
        g = pickle.load(f)

    launch_analysis(dataset=central_dataset, mtg=g, output_dir=working_dir)


if __name__ == '__main__':
    analyze_multiple_scenarios(scenarios_set=0)
