import os
import numpy as np
import pickle

import matplotlib.pyplot as plt
import xarray as xr

from root_cynaps.tools_output import plot_xr, plot_N, global_state_extracts, global_flow_extracts



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

    g_name = [name for name in os.listdir(working_dir) if ".pckl" in name][0]
    with open(working_dir + '/' + g_name, 'rb') as f:
        g = pickle.load(f)

    # Plotting global outputs
    # print("PLOTTING GLOBAL PROPERTIES...")
    # plot_xr(datasets=datasets, selection=list(global_state_extracts.keys()), supplementary_legend=supplementary_legend)
    # plot_xr(datasets=datasets, selection=list(global_flow_extracts.keys()), supplementary_legend=supplementary_legend)
    # plt.ion()

    # For some reason, dataset should be loaded before umap
    from tools import STM_analysis
    # Running STM sensitivity analysis
    STM_analysis.run(file=central_dataset)

    # TODO Plotting target local values

    # TODO Plotting values projected on architecture


def plot_multiple_scenarios(g, datasets, supplementary_legend, set_name, time_steps=[]):

    # plot global properties


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
    merged_dataset.to_netcdf("C:/Users/tigerault/pp/root_cynaps/simulations/running_scenarios/outputs/tests.nc")
    #print(merged_dataset)

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
