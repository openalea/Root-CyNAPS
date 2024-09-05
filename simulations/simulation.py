# Public packages
import os, sys, time
import multiprocessing as mp
# Model packages
from root_cynaps.root_cynaps import Model
# Utility packages
from log.logging import Logger
from initialize.initialize import MakeScenarios as ms
from analyze.analyze import analyze_data


def single_run(scenario, outputs_dirpath="outputs", simulation_length=2500, echo=True, log_settings={}):
    root_cynaps = Model(time_step=3600, **scenario)

    logger = Logger(model_instance=root_cynaps, outputs_dirpath=outputs_dirpath, 
                    time_step_in_hours=1, logging_period_in_hours=1,
                    echo=echo, auto_camera_position=True, **log_settings)
    
    try:
        for _ in range(simulation_length):
            # Placed here also to capture mtg initialization
            logger()
            logger.run_and_monitor_model_step()
            #root_cynaps.run()

    except (ZeroDivisionError, KeyboardInterrupt):
        logger.exceptions.append(sys.exc_info())

    finally:
        logger.stop()
        analyze_data(scenarios=[os.path.basename(outputs_dirpath)], outputs_dirpath="outputs", target_properties=None, **log_settings)
        


def simulate_scenarios(scenarios, simulation_length=2500, echo=True, log_settings={}):
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
                                                      log_settings=log_settings))
        p.start()
        processes.append(p)

        
if __name__ == "__main__":
    #scenarios = ms.from_table(file_path="inputs/Scenarios_24_06.xlsx", which=["Input_RSML_D9", "Input_RSML_D11", "Input_RSML_D13"])
    scenarios = ms.from_table(file_path="inputs/Scenarios_24_06.xlsx", which=["Input_RSML_D13"])
    simulate_scenarios(scenarios, simulation_length=5, log_settings=Logger.heavy_log)