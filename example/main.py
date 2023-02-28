from rhizodep.simulation_no_C import N_simulation


def main(init, n, time_step):
    N_simulation(init=init, n=n, time_step=time_step)


main(init='inputs/root00478.pckl', n=20, time_step=3600)
