from rhizodep.simulation_no_C import N_simulation


def main(init, n, time_step):
    N_simulation(init=init, n=n, time_step=time_step, discrete_vessels=True, plotting=False, logging=True)


main(init='inputs/root00020.pckl', n=40, time_step=3600)
