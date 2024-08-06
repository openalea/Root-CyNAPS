from initialize.initialize import MakeScenarios as ms
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openalea.mtg.traversal import pre_order2
from log.visualize import plot_mtg_alt
import pyvista as pv
import numpy as np


def test_mtg_integrity():
    scenarios = ms.from_table(file_path="inputs/Scenarios_24_06.xlsx", which=["Input_RSML"])

    g = scenarios["Input_RSML"]["input_mtg"]["root_mtg_file"]
    #mtg_2D_scatter(g)

    mtg_manual_graph_checker(g)

    #mtg_pyvista_plotter(g)

def mtg_pyvista_plotter(g):
    plotter = pv.Plotter(off_screen=False, window_size=[1920, 1080], lighting="three lights")
    plotter.set_background("brown")

    step_back_coefficient = 0.5
    camera_coordinates = (step_back_coefficient, 0., 0.)
    move_up_coefficient = 0.1
    horizontal_aiming = (0., 0., 1.)
    collar_position = (0., 0., -move_up_coefficient)
    plotter.camera_position = [camera_coordinates,
                                    collar_position,
                                    horizontal_aiming]

    root_system_mesh, color_property = plot_mtg_alt(g, cmap_property="z2", flow_property=False)
    plotter.add_mesh(root_system_mesh, cmap="jet", clim=[1e-10, 6e-9], show_edges=False, log_scale=True)
    plotter.show(interactive_update=False)
    


def compute_distance_between_vertices(v1, v2):
    try:
        return np.sqrt((v1.x1 - v2.x1)**2 + (v1.y1 - v2.y1)**2 + (v1.z1 - v2.z1)**2)
    except AttributeError:
        return None
    
def mtg_manual_graph_checker(g):
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    for vid in pre_order2(g, root):
        n = g.node(vid)
        print(n.label, vid, n.children(), compute_distance_between_vertices(n, n.parent()))
        input()


def mtg_2D_scatter(g, dot_size=2):
    fig, ax = plt.subplots(1, 2)
    Xs, Ys, Zs = [], [], []
    for vid in g.vertices():
        n = g.node(vid)
        if n.x2 and n.y2 and n.z2:
            Xs.append(n.x2)
            Ys.append(n.y2)
            Zs.append(-n.z2)
    ax[0].scatter(Xs, Zs, s=dot_size)
    ax[1].scatter(Ys, Zs, s=dot_size)
    fig.savefig('outputs/mtg_graph.png')  # Save as PNG
    plt.close()

if __name__ == "__main__":
    test_mtg_integrity()