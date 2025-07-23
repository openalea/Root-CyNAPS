import os
import numpy as np

# Utility packages
from analyze.analyze import analyze_data, test_output_range
from log.visualize import post_compress_gltf, custom_colorbar


if __name__ == '__main__':
        
    for scenario_name in ["RC_ref_50"]:
    # for scenario_name in ["RC_ref_20"]:
    # for scenario_name in [f"RC_ref_{10*(k+1)}" for k in range(6)]:
        # output_path = os.path.join("outputs", "fig_visuals")
        output_path = os.path.join("outputs", "fig_visuals_bis")
        # output_path = os.path.join("outputs", "fig_batch")

        target_folder_key = "RootCyNAPS_0"
        
        # target_concentrations = np.logspace(0, 4, 5) * 5e-3
        target_concentrations = [5e-1]

        # conversion_factor = 3600 / 100 * 1e9 # for water µL.cm-1.h-1
        # vmin, vmax = 3e-10 * conversion_factor, 6e-10 * conversion_factor # for water
        # conversion_factor = 3600 / 100 * 1e9 # for N uptake and AA exudation nmol.cm-1.h-1
        # vmin, vmax = 1e-11 * conversion_factor, 1.5e-10 * conversion_factor # for Nm uptake 1e-11, 1.5e-10
        # vmin, vmax = 1e-12 * conversion_factor, 4e-11 * conversion_factor # for AA exudation

        # custom_colorbar(folderpath=output_path, label="manual", vmin=vmin, vmax=vmax, 
        #                                     colormap="cool", vertical=True, log_scale=False, filename=f"manual_colorbar_aa.png")

        for concentration in target_concentrations:
            analyze_data(scenarios=[f"{str(scenario_name)}_{concentration:.2e}"], outputs_dirpath=output_path, target_folder_key=target_folder_key,
                            inputs_dirpath="inputs",
                            on_sums=False,
                            on_performance=False,
                            animate_raw_logs=True,
                            target_properties=None
                            )
    