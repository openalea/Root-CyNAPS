from rhizodep.simulation_no_C import N_simulation


def main(init, n, time_step):
    N_simulation(init=init, n=n, time_step=time_step, discrete_vessels=True, plotting=False)


main(init='inputs/root00478.pckl', n=60, time_step=3600)
