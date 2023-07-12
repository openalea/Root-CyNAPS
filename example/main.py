import os
import timeit

from rhydromin.simulation_no_C import N_simulation


def main(init, n, time_step):
    N_simulation(init=init, n=n, time_step=time_step, discrete_vessels=True, plantgl=False, plotting_2D=True, plotting_STM=False, logging=True)

# This step is essential to ensure a systematic reference of subdirectories, at least for windows paths
os.chdir(os.getcwd()[:os.getcwd().find("example")-1])
main(init='example/inputs/root00020.pckl', n=24, time_step=3600)
