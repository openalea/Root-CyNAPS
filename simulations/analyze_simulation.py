# Utility packages
from analyze.analyze import analyze_data, test_output_range


if __name__ == '__main__':

    # simulation_block_names = ["reference", "unitary_N", "unitary_W", "cross_CN"]

    scenarios = ["Input_RSML_R1_D13", "Input_RSML_R2_D13", "Input_RSML_R4_D13", 
                 "Input_RSML_R1_D11", "Input_RSML_R2_D11", "Input_RSML_R3_D11", "Input_RSML_R4_D11", 
                 "Input_RSML_R1_D09", "Input_RSML_R2_D09", "Input_RSML_R3_D09", "Input_RSML_R4_D09",
                 "Input_RSML_R1_D07", "Input_RSML_R2_D07","Input_RSML_R3_D07",
                 "Input_RSML_R1_D05","Input_RSML_R2_D05","Input_RSML_R3_D05","Input_RSML_R4_D05"]

    # for block_name in simulation_block_names:
    #     block_scenarios = [block_name + "_" + scenario for scenario in scenarios]
    #     analyze_data(scenarios=block_scenarios, outputs_dirpath="outputs", on_sums=False, animate_raw_logs=True, subdir_custom_name=block_name)

    scenarios = ["Input_RSML_R4_D13"]
    #analyze_data(scenarios=scenarios, outputs_dirpath="outputs", on_sums=False, animate_raw_logs=True)

    test_output_range(scenarios=scenarios, outputs_dirpath="outputs", test_file_dirpath="inputs/outputs_validation_root_cynaps_v0.xlsx")
        