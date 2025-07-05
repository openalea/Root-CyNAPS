import os
import numpy as np

# Utility packages
from analyze.analyze import analyze_data, test_output_range
from log.visualize import post_compress_gltf


if __name__ == '__main__':
        
    for scenario_name in ["RC_ref_50"]:
        scenarios = []

        output_path = os.path.join("outputs", "fig_7.5")

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
    