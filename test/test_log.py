import pickle

from root_cynaps.root_cynaps import Model

def test_log():
    with open('inputs/root00119.pckl', 'rb') as f:
        g = pickle.load(f)
    
    root_cynaps = Model(g=g, time_step=3600)
    mtg = root_cynaps.g

    for step in range(10):
        assert root_cynaps.root_nitrogen.xylem_water == root_cynaps.root_water.xylem_water == mtg.properties()["xylem_water"]
        root_cynaps.run()

test_log()
