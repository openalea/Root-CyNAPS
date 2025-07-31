# Public packages
import multiprocessing as mp
import time

# Model packages
import openalea.rootcynaps
from openalea.rootcynaps.soon_public_packages.static_soil_assembly import StaticSoilAssembly
from openalea.rootcynaps import RootCyNAPS

# Utility packages
from openalea.fspm.utility.scenario import MakeScenarios as ms
from openalea.fspm.utility.writer import Logger
from openalea.fspm.utility.plot import analyze_data
from openalea.metafspm.scene_wrapper import play_Orchestra


if __name__ == '__main__':
    # scenarios = ms.from_table(file_path="inputs/Scenarios_25_07_02.xlsx", which=[f"RC_ref_{5*(k+1)}" for k in range(12)])
    # scenarios = ms.from_table(file_path="inputs/Scenarios_25_07_02.xlsx", which=[f"RC_ref_{5 + 10*(k)}" for k in range(6)])
    # scenarios = ms.from_table(file_path="inputs/Scenarios_25_07_02.xlsx", which=[f"RC_ref_{10*(k+1)}" for k in range(6)])
    scenarios = ms.from_table(file_path="inputs/Scenarios_25_07_02.xlsx", which=["RC_ref_50"])
    custom_output_folder = "outputs/fig_standalone"
    # custom_output_folder = "outputs/fig_visuals_aa_exudation"
    # custom_output_folder = "outputs/fig_visuals_water"
    # custom_output_folder = "outputs/fig_batch_net_N_uptake"

    scene_xrange = 0.15
    scene_yrange = 0.15
    sowing_density = 1
    environment_models_number = 1
    subprocesses_number = int(max(scene_xrange * scene_yrange * sowing_density, 1)) + environment_models_number
    parallel_development = 1 # To keep room in CPUs if launching dev simulations in parallel on the machine
    max_processes = mp.cpu_count() - subprocesses_number - parallel_development - 1 # -1 for the main process

    # target_concentrations = np.logspace(0, 4, 5) * 5e-3
    # explored_space = np.logspace(0, 4, 9) * 5e-3
    # target_concentrations = [explored_space[i] for i in range(len(explored_space)-1) if i % 2 == 1]
    target_concentrations = [5e-1]    

    parallel = False
    active_processes = 0 
    processes = []

    for scenario_name, scenario in scenarios.items():
        
        if parallel:
            for concentration in target_concentrations:
                scenario["parameters"]["soil_model"]["soil"]["dissolved_mineral_N"] = 2.958e-6 * concentration

                # Main process creation part

                # Wait until there is a free slot
                while True:
                    # Remove any finished processes and join them to release resources
                    alive_processes = []
                    for proc in processes:
                        if proc.is_alive():
                            alive_processes.append(proc)
                        else:
                            proc.join()
                    processes = alive_processes

                    if len(processes) < max_processes:
                        break
                    time.sleep(1)

                print("")
                print(f'### Launching {scenario_name} over already {active_processes} processes running ###')
                print("")
                active_processes += subprocesses_number
                    
                current_scenario_name = f"{str(scenario_name)}_{concentration:.2e}"

                p = mp.Process(target=play_Orchestra, kwargs=dict(scene_name=current_scenario_name, output_folder=custom_output_folder, plant_models=[RootCyNAPS], plant_scenarios=[scenario], 
                                                                soil_model=StaticSoilAssembly, soil_scenario=scenario,
                                                                translator_path=openalea.rootcynaps.__path__[0],
                                                                logger_class=Logger, log_settings=Logger.light_log,
                                                                scene_xrange=scene_xrange, scene_yrange=scene_yrange, sowing_density=sowing_density,
                                                                time_step=3600, n_iterations=24))
                
                p.start()
                processes.append(p)

        else:
            for concentration in target_concentrations:
                scenario["parameters"]["root_cynaps"]["roots"]["dissolved_mineral_N"] = 5e-7 * concentration / 1e-1
                
                current_scenario_name = f"{str(scenario_name)}_{concentration:.2e}"
                print(openalea.rootcynaps.__path__[0])
                play_Orchestra(scene_name=current_scenario_name, output_folder=custom_output_folder, plant_models=[RootCyNAPS], plant_scenarios=[scenario], 
                                    soil_model=StaticSoilAssembly, soil_scenario=scenario,
                                    translator_path=openalea.rootcynaps.__path__[0],
                                    logger_class=Logger, log_settings=Logger.light_log,
                                    scene_xrange=scene_xrange, scene_yrange=scene_yrange, sowing_density=sowing_density,
                                    time_step=3600, n_iterations=24)

                target_folder_key = "RootCyNAPS_0"

                analyze_data(scenarios=[current_scenario_name], outputs_dirpath=custom_output_folder, target_folder_key=target_folder_key,
                                inputs_dirpath="inputs",
                                on_sums=True,
                                on_performance=False,
                                animate_raw_logs=False,
                                target_properties=None
                                )
                
                
     # After all tasks started, join any remaining processes
    for proc in processes:
        proc.join()