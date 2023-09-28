import os
import shutil

from root_cynaps.simulation_no_C import N_simulation

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

def main(output_path, hexose_decrease_rate, z_soil_Nm_max, init='root00020.pckl', n=144, time_step=3600, echo=False):
    """
    :Parameters
    Every unchanged argument should be placed as default parameter for ease of multiprocessing
    """
    # This step is essential to ensure a systematic reference of subdirectories
    current_file_dir = os.path.dirname(__file__)
    if echo:
        plantgl=False
        plotting_2D = True
        plotting_STM = False
    else:
        plantgl = False
        plotting_2D = False
        plotting_STM = False

    N_simulation(output_path, hexose_decrease_rate, z_soil_Nm_max, current_file_dir=current_file_dir, init=init, n=n, time_step=time_step, echo=echo,
                 plantgl=plantgl, plotting_2D=plotting_2D, plotting_STM=plotting_STM, logging=True)


if __name__ == '__main__':
    previous_outputs_clearing()
    main(output_path=os.path.dirname(__file__) + '/outputs', hexose_decrease_rate=3, z_soil_Nm_max=0, echo=True)
