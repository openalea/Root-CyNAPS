import pickle

from root_cynaps.root_cynaps import Model

def test_import():
    with open('inputs/root00119.pckl', 'rb') as f:
        g = pickle.load(f)
    
    root_cynaps = Model(g=g, time_step=3600)