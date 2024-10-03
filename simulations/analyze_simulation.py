# Utility packages
from analyze.analyze import analyze_data


if __name__ == '__main__':

    scenarios = ["Input_RSML_R1_D13", "Input_RSML_R2_D13", "Input_RSML_R4_D13", 
                 "Input_RSML_R1_D11", "Input_RSML_R2_D11", "Input_RSML_R3_D11", "Input_RSML_R4_D11", 
                 "Input_RSML_R1_D09", "Input_RSML_R2_D09", "Input_RSML_R3_D09", "Input_RSML_R4_D09",
                 "Input_RSML_R1_D07", "Input_RSML_R2_D07","Input_RSML_R3_D07",
                 "Input_RSML_R1_D05","Input_RSML_R2_D05","Input_RSML_R3_D05","Input_RSML_R4_D05"]
    scenarios = ["Input_RSML_R4_D13"]
    
    analyze_data(scenarios=scenarios, outputs_dirpath="outputs", animate_raw_logs=True)
