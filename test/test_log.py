import pickle

from root_cynaps.root_cynaps import Model

def test_log():
    with open('inputs/root00119.pckl', 'rb') as f:
        g = pickle.load(f)
    
    root_cynaps = Model(g=g, time_step=3600)
    mtg = root_cynaps.g

    for step in range(10):
        print(mtg.properties()["total_struct_mass"])
        root_cynaps.run()

test_log()
