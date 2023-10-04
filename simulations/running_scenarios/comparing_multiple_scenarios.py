import os
import numpy as np
import pickle

import matplotlib.pyplot as plt
import xarray as xr

from root_cynaps.tools_output import plot_xr, plot_N, global_state_extracts, global_flow_extracts


def analyze_multiple_scenarios(scenarios_set):
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

    mtg_path = root_path + "/outputs/" + working_dir[scenarios_set] + '/' + list_dir[0]
    g_name = [name for name in os.listdir(mtg_path) if ".pckl" in name][0]
    with open(root_path + "/outputs/" + working_dir[scenarios_set] + '/' + list_dir[0] + '/' + g_name, 'rb') as f:
        g = pickle.load(f)

    plot_multiple_scenarios(g=g, datasets=time_datasets, supplementary_legend=legend, set_name=working_dir[scenarios_set], time_steps=[100, 120])


def plot_multiple_scenarios(g, datasets, supplementary_legend, set_name, time_steps=[]):

    # plot global properties
    print("PLOTTING GLOBAL PROPERTIES...")
    plot_xr(datasets=datasets, selection=list(global_state_extracts.keys()), supplementary_legend=supplementary_legend)
    plot_xr(datasets=datasets, selection=list(global_flow_extracts.keys()), supplementary_legend=supplementary_legend)
    plt.ion()

    print("PROCESSING SPATIALIZED DATA...")
    for k in range(len(supplementary_legend)):
        supplementary_legend[k] = [float(k) for k in eval(supplementary_legend[k])]

    # Finally, do regression along the axes and plot p_value in plantgl

    scenario_variables = []
    for k in range(set_name.count(" X ")):
        cut = set_name.find(" X ")
        scenario_variables += [set_name[:cut]]
        set_name = set_name[cut + 3:]
    scenario_variables += [set_name[:set_name.find(" ")]]

    # plot spatialized property correlation with scenarios' variables on plantgl
    prop = "import_Nm"
    for d in range(len(datasets)):
        coordinates = dict(zip(scenario_variables, supplementary_legend[d]))
        datasets[d] = datasets[d].assign_coords(coords=coordinates).expand_dims(dim=dict(zip(scenario_variables, [1 for k in scenario_variables])))
    print("Merging datasets from different scenarios...")
    merged_dataset = xr.merge(datasets)
    for time_step in time_steps:
        regression_extract = getattr(merged_dataset, prop).sel(t=time_step)
        print(f"Regression over {scenario_variables} values for {prop}")
        # regression_results = regression_extract.polyfit(dim=scenario_variables[0], deg=1, full=True, cov=True)
        regression_results = regression_extract.curvefit(coords=scenario_variables, func=multilinear_f, param_names=["P_" + name for name in scenario_variables] + ["origin_value"])
        # TODO Make better use of the covariance matrix.
        regression_coefficients = regression_results.sel(param="P_hexose_decrease_rate")["curvefit_coefficients"].to_dict()
        # Note here that we normalize by dividing by root segment structural mass to avoid under representation of young short segments
        coefficients_dict = {k: v/(g.properties()["struct_mass"][k]) for k, v in zip(regression_coefficients["coords"]["vid"]["data"], regression_coefficients["data"])}

        print("PLOTTING REGRESSION RESULTS WITH PLANTGL...")
        g.properties()["coef_hexose"] = coefficients_dict

        # legend plot TODO : side to side plots
        fig, axs = plt.subplots(2, 1)
        fig.subplots_adjust(left=0.2, bottom=0.2)
        plot_N(g=g, p=["coef_hexose"], axs=axs)

    # plot local properties once differences are identified spatially
    # TODO

    # Wait until user ends
    plt.show()
    input("end?")


# Tool functions
def multilinear_f(*args):
    """
    Description : This function with variable argument length returns the linear combination of input variables
    """
    arguments = args[0]
    parameters = args[1:]
    origin_value = parameters[-1]
    products = []
    for a in range(len(arguments)):
        products += [[x*parameters[a] for x in arguments[a]]]
    summed_product = []
    for rep in range(len(arguments[0])):
        s = 0
        for d in range(len(arguments)):
            s += products[d][rep]
        s += origin_value
        summed_product += [s]

    return np.array(summed_product)


if __name__ == '__main__':
    analyze_multiple_scenarios(scenarios_set=0)
