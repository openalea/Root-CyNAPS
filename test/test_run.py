import pickle

from root_cynaps.root_cynaps import Model

def test_run():
    with open('inputs/root00119.pckl', 'rb') as f:
        g = pickle.load(f)
    
    root_cynaps = Model(g=g, time_step=3600)

    for step in range(10):
        root_cynaps.run()

test_run()