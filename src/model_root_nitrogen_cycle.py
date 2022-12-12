# Loading packages:
import os
import numpy as np
import pickle
from openalea.mtg import *
from openalea.mtg.traversal import pre_order, post_order
import src.parameters as param


########################################################################################################################

def mineral_nitrogen_uptake(g):
    """
    Active mineral nitrogen uptake from soil to symplasm of each root segment in g
    :param g: the MTG to be read
    :param time_step: the time step in seconds (default
    :return: g, the new updated MTG
    """

    # We define "root" as the starting point of the loop below:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    # We travel in the MTG from the root tips to the base:
    for vid in post_order(g, root):
        # We define the current root element as n:
        n = g.node(vid)
        # We define the amount of nitrogen that has been taken (mol of N) in a perfectly stupid way:
        n.influx_mineral_nitrogen = (n.soil_mineral_nitrogen * param.influx_mineral_nitrogen_max / (
                    n.soil_mineral_nitrogen + param.affinity_mineral_nitrogen)) * 2 * np.pi * n.radius * n.length
    return g


def root_mineral_nitrogen_update(g, time_step=3600):
    """
    Local mineral nitrogen content update
    :param time_step:
    :param g: the MTG to be read
    :return:
    """

    # We define "root" as the starting point of the loop below:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    # We travel in the MTG from the root tips to the base:
    for vid in post_order(g, root):
        # We define the current root element as n:
        n = g.node(vid)
        # We define the amount of nitrogen that has been taken (mol of N) in a perfectly stupid way:
        n.root_mineral_nitrogen += time_step * (n.influx_mineral_nitrogen)
    return g


########################################################################################################################

# INITIALIZATION:
#################

# We define the path of the directory:
my_path = r'C:\Users\tigerault\PycharmProjects\cnrhizowheat\simulations\fixed_architechure\input\MTG_files'
os.chdir(my_path)
# We open the MTG file that has already been prepared and contains all RhizoDep's variables:
f = open('root00119.pckl', 'rb')
# We define "g" as the new MTG that has been loaded:
g = pickle.load(f)
f.close()


# We add and set MTG property, now expressed as constant for initialisation
# but that will be further modified by other processes.

# We define "root" as the starting point of the loop below:
root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
root = next(root_gen)
#  We travel in the MTG from the root tips to the base:
for vid in post_order(g, root):
    # We define the current root element as n:
    n = g.node(vid)
    # We set new properties related to Nitrogen cycle
    n.soil_mineral_nitrogen = 0.1
    n.root_mineral_nitrogen = 0.1
    n.influx_mineral_nitrogen = 0

# # We run a simulation over 10 time steps:
n_steps = 10
# # We initialize the time at 0:
time = 0
# # We calculate things for each time step:
for step in range(0, n_steps):
    # We increment the time:
    time += 1
    print("Calculating at time t =", time, 'h...')
    # We call the function nitrogen uptake for this time step:
    mineral_nitrogen_uptake(g)
    root_mineral_nitrogen_update(g)
    print(g.node(1).root_mineral_nitrogen)
#
# print('Simulation is done!')
