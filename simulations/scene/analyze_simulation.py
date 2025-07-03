import os
import numpy as np

# Utility packages
from analyze.analyze import analyze_data, test_output_range
from log.visualize import post_compress_gltf


if __name__ == '__main__':

    # scenarios = ["Drew_1975_low", "Drew_1975_1"]
    #scenarios = ["Drew_1975_low"]
    # scenarios = ["RC_ref_30D_debug"]
    # target_days = [ 5, 7, 10, 20, 30, 40, 50, 60] #, 70, 80, 90, 100]
    # target_days = np.arange(10, 61, 1)
    # target_days = [10, 20, 30, 40, 50, 60]
    
    # scenarios = [f"RC_ref_high_{day}D" for day in target_days]
    # scenarios = [f"RC_ref_{day}D" + "_images" for day in target_days]
    # scenarios = [f"RC_no_hair_{day}D" for day in target_days]
    # scenarios = [f"RC_ref_{day}D" for day in target_days] + [f"RC_no_hair_{day}D" for day in target_days]

    # for n_scenario in ["RC_ref", "RC_ref_0.01",	"RC_ref_0.05", "RC_ref_0.5", "RC_ref_5", "RC_ref_50"]:
    # for n_scenario in ["RC_ref"]:
    # # for n_scenario in ["RC_ref_big_lats"]:
    #     # target_days = [10, 20, 30, 40, 50]
    #     target_days = [50]
    #     target_concentration = "5.00e-01"
    #     # target_smax = np.logspace(0, 4, 11) * 1e-9
    #     target_smax = [5e-6]

    #     scenarios = [f"{n_scenario}_{target_concentration}_{day}D" for day in target_days]
    #     # scenarios = [f"{n_scenario}_{target_concentration}_{target_days[0]}D_{smax:.2e}max" for smax in target_smax]

    #     # output_path = "outputs"
    #     output_path = os.path.join("outputs", "batch_fig7")
        
    #     #output_path = "C:/Users/tigerault/OneDrive - agroparistech.fr/Thesis/Sujet/Modelling/saved_scenarios/05-06_hairless_tests"
    #     # output_path = "C:/Users/tigerault/OneDrive - agroparistech.fr/Thesis/Sujet/Modelling/saved_scenarios/01-06_ISRR 2024"

    #     # test_output_range(scenarios=scenarios, outputs_dirpath="outputs", test_file_dirpath="inputs/outputs_validation_root_cynaps_V0.xlsx")

    #     # post_compress_gltf(os.path.join(output_path, scenarios[0], "root_images"))

    #     analyze_data(scenarios=scenarios, outputs_dirpath=output_path, inputs_dirpath="inputs",
    #                     on_sums=False,
    #                     on_performance=False,
    #                     animate_raw_logs=True,
    #                     target_properties=None
    #                     )
        
    for scenario_name in ["RC_ref_50"]:
        scenarios = []

        output_path = os.path.join("outputs", "fig_7.4")

        # Use bellow for debug
        # output_path = os.path.join("Root_BRIDGES", "simulations", "scene", "outputs", "fig_7.3")
        target_folder_key = "RootCyNAPS_0"
        
        target_days = [50]
        target_concentrations = np.logspace(0, 4, 5) * 5e-3
        target_concentrations = [5e-1]
        
        for day in target_days:
            for concentration in target_concentrations:
                analyze_data(scenarios=[f"{str(scenario_name)}_{concentration:.2e}"], outputs_dirpath=output_path, target_folder_key=target_folder_key,
                                inputs_dirpath="inputs",
                                on_sums=True,
                                on_performance=False,
                                animate_raw_logs=False,
                                target_properties=None
                                )
        
        # In the end put the system to sleep, Windows only
        # os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
    