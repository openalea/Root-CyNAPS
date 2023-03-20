from rhizodep.simulation_no_C import N_simulation
from rhizodep.converter import root_shoot_converter

def main(init, n, time_step):
    N_simulation(init=init, n=n, time_step=time_step, outside_flows=root_shoot_converter(model_path="cn-wheat"))


main(init='inputs/root00478.pckl', n=60, time_step=3600)
