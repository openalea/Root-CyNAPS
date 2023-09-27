import os
import shutil
import time
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import product
import multiprocessing as mp
import tqdm

from simulations.running_example.main import main
from simulations.running_scenarios.comparing_multiple_scenarios import plot_multiple_scenario


def run_one_scenario(scenario: dict = {}):
    main(**scenario)
    return


def previous_outputs_clearing():
    root_path = os.path.dirname(__file__)
    try:
        # We remove all files and subfolders:
        try:
            shutil.rmtree(root_path + '/outputs')
            print("Deleted the 'outputs' folder...")
            print("Creating a new 'outputs' folder...")
            os.mkdir('outputs')
        except OSError:
            print("Creating a new 'outputs' folder...")
            os.mkdir('outputs')
    except OSError as e:
        print("An error occured when trying to delete the output folder: %s - %s." % (e.filename, e.strerror))


def function_from_text(function, argument):
    y = []
    for x in argument:
        y += [eval(function)]
    return y


def run_multiple_scenarios(scenarios_list="scenarios_variables.xlsx"):
    # READING SCENARIO INSTRUCTIONS:
    root_path = os.path.dirname(__file__)

    # FROM A CSV FILE where scenarios are present in different lines:
    if os.path.splitext(scenarios_list)[1] == ".csv":
        # We read the data frame containing the different scenarios to be simulated:
        print("Loading the instructions of scenarios...")
        scenarios_df = pd.read_csv(root_path + '/inputs/' + scenarios_list)
        # We copy the list of scenarios' properties in the 'outputs' directory:
        scenarios_df.to_csv(root_path + '/outputs/scenarios_list.csv', na_rep='NA', index=False, header=True)
        # We record the number of each scenario to be simulated:
        print(scenarios_df['num'])

    # FROM AN EXCEL FILE where scenarios are present in different columns:
    elif os.path.splitext(scenarios_list)[1] == ".xlsx":
        # We read the data frame containing the different scenarios to be simulated:
        print("Loading the instructions of scenario(s)...")
        scenarios_df = pd.read_excel(os.path.join('inputs', scenarios_list), sheet_name="scenarios_variables")

        start_time = datetime.now().strftime("%y.%m.%d_%H.%M")
        folder_name = " X ".join(list(scenarios_df["variable_name"])) + " S " + start_time
        print("Launching combinations for " + folder_name)
        os.mkdir(root_path + '/outputs/' + folder_name)

        distribution = []
        for sc in range(scenarios_df.shape[0]):
            num = scenarios_df["num"][sc]
            l_distribution = np.linspace(start=scenarios_df["start"][sc], stop=scenarios_df["stop"][sc], num=num)
            function = scenarios_df["distribution_function"][sc]
            distribution += [function_from_text(function=function, argument=l_distribution)]

        scenarios = []
        for combination in list(product(*distribution)):
            scenario = {}
            names = scenarios_df["variable_name"]
            file_name = ["{:.2e}".format(k) for k in combination]
            path = root_path + '/outputs/' + folder_name + '/' + str(file_name)
            os.mkdir(path)
            for k in range(len(names)):
                scenario.update({"output_path": path, names[k]: combination[k]})
            scenarios += [scenario]

    # We record the starting time of the simulation:
    t_start = time.time()
    # We look at the maximal number of parallel processes that can be run at the same time:
    num_processes = mp.cpu_count()
    p = mp.Pool(num_processes)

    # We run all scenarios in parallel:

    # WATCH OUT: p.map does not allow to have multiple arguments in the function to be run in parallel!!!
    # Arguments of 'run_one_scenario' should be modifed directly in the default parameters of the function.
    print("=============== Progress ================")
    print("|                                       |")
    for _ in tqdm.tqdm(p.imap_unordered(run_one_scenario, scenarios), total=len(scenarios)):
        pass

    p.terminate()
    p.join()

    # We indicate the total time the simulations took:
    t_end = time.time()
    tmp = (t_end - t_start) / 60.
    print("|                                       |")
    print("=========================================")
    print("Multiprocessing took %4.3f minutes!" % tmp)


if __name__ == '__main__':
    previous_outputs_clearing()
    run_multiple_scenarios()
    plot_multiple_scenario(scenarios_set=0)
