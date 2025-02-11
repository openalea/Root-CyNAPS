# Public packages
import os, sys, time
import multiprocessing as mp
# Model packages
from root_cynaps.root_cynaps import Model
# Utility packages
from log.logging import Logger
from initialize.initialize import MakeScenarios as ms
from analyze.analyze import analyze_data


def single_run(scenario, outputs_dirpath="outputs", simulation_length=2500, echo=True, log_settings={}, analyze=True):
    root_cynaps = Model(time_step=3600, **scenario)

    logger = Logger(model_instance=root_cynaps, components=root_cynaps.components,
                    outputs_dirpath=outputs_dirpath,
                    time_step_in_hours=1, logging_period_in_hours=6,
                    echo=echo, static_mtg=True, **log_settings)
    
    try:
        for _ in range(simulation_length):
            # Placed here also to capture mtg initialization
            logger.run_and_monitor_model_step()
            #logger()
            #root_cynaps.run()

    except (ZeroDivisionError, KeyboardInterrupt):
        logger.exceptions.append(sys.exc_info())

    finally:
        logger.stop()
        if analyze:
            analyze_data(scenarios=[os.path.basename(outputs_dirpath)], outputs_dirpath="outputs", target_properties=None, **log_settings)
        


def simulate_scenarios(scenarios, simulation_length=2500, echo=True, log_settings={}, analyze=True):
    processes = []
    max_processes = mp.cpu_count()
    for scenario_name, scenario in scenarios.items():
        
        while len(processes) == max_processes:
            for proc in processes:
                if not proc.is_alive():
                    processes.remove(proc)
            time.sleep(1)
        
        print(f"[INFO] Launching scenario {scenario_name}...")
        p = mp.Process(target=single_run, kwargs=dict(scenario=scenario, 
                                                      outputs_dirpath=os.path.join("outputs", str(scenario_name)),
                                                      simulation_length=simulation_length,
                                                      echo=echo,
                                                      log_settings=log_settings,
                                                      analyze=analyze))
        p.start()
        processes.append(p)

        
if __name__ == "__main__":
    print("Starting simulation in 2 seconds, use Ctrl+C to cancel !")
    time.sleep(2)

    simulation_block = {"reference": {"patch_depth_mineralN": None, "patch_depth_water_moisture": None},
                        "unitary_N": {"patch_depth_mineralN": 0., "patch_depth_water_moisture": None},
                        "unitary_W": {"patch_depth_mineralN": None, "patch_depth_water_moisture": 0.},
                        "cross_CN": {"patch_depth_mineralN": 0., "patch_depth_water_moisture": 0.}
                        }

    # scenarios = ms.from_table(file_path="inputs/Scenarios_24_09_22.xlsx", which=[
    #            "Input_RSML_R1_D13", "Input_RSML_R2_D13", "Input_RSML_R4_D13", 
    #              "Input_RSML_R1_D11", "Input_RSML_R2_D11", "Input_RSML_R3_D11", "Input_RSML_R4_D11", 
    #              "Input_RSML_R1_D09", "Input_RSML_R2_D09", "Input_RSML_R3_D09", "Input_RSML_R4_D09",
    #              "Input_RSML_R1_D07", "Input_RSML_R2_D07","Input_RSML_R3_D07",
    #              "Input_RSML_R1_D05","Input_RSML_R2_D05","Input_RSML_R3_D05","Input_RSML_R4_D05"])

    scenarios = ms.from_table(file_path="inputs/Scenarios_24_09_22.xlsx", which=["Input_RSML_R4_D13"])
    
    # senarios_blocks = ms.variate_scenario_sets(scenarios, simulation_block)

    # for scenarios_block in senarios_blocks:
    #     simulate_scenarios(scenarios_block, simulation_length=48, log_settings=Logger.heavy_log, analyze=False)

    simulate_scenarios(scenarios, simulation_length=48, log_settings=Logger.light_log, analyze=False)