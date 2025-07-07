# Public packages
import os, sys, time
import pyvista as pv
from openalea.mtg.traversal import post_order

# Utility packages
from log.visualize import plot_mtg_alt
from analyze.analyze import add_root_order_when_branching_is_wrong
from initialize.initialize import MakeScenarios as ms


class VertexPicker:
    def __init__(self, g, target_property="radius"):
        self.g = g
        self.props = g.properties()
        self.target_property = target_property

    # Define the callback for when an object is picked
    def __call__(self, picked_coordinates):
        x, y, z = picked_coordinates
        distances = {}
        for vid in self.g.vertices():
            if vid != 0:
                distance = self.compute_vertex_distance(x, y, z, self.props["x2"][vid], self.props["y2"][vid], self.props["z2"][vid])
                distances[vid] = distance

        picked = min(distances, key=distances.get)
        print(f"Picked vertex {picked} (+{self.g.children(picked)} -{self.g.parent(picked)}), distance_from_tip = {self.props['distance_from_tip'][picked]}, {self.target_property} = {self.props[self.target_property][picked]}")

    def compute_vertex_distance(self, x1, y1, z1, x2, y2, z2):
        return ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5
    

def update_distance_from_tip(g):
    """
    The function "distance_from_tip" computes the distance (in meter) of a given vertex from the apex
    of the corresponding root axis in the MTG "g" based on the properties "length" of all vertices.
    Note that the dist-to-tip of an apex is defined as its length (and not as 0).
    :return: the MTG with an updated property 'distance_from_tip'
    """

    # We initialize an empty dictionary for to_tips:
    to_tips = {}
    # We use the property "length" of each vertex based on the function "length":
    length = g.property('length')

    # We define "root" as the starting point of the loop below:
    root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
    root = next(root_gen)

    # We travel in the MTG from the root tips to the base:
    for vid in post_order(g, root):
        # We define the current root element as n:
        n = g.node(vid)
        # We define its direct successor as son:
        son_id = g.Successor(vid)
        son = g.node(son_id)

        # We try to get the value of distance_from_tip for the neighbouring root element located closer to the apex of the root:
        try:
            # We calculate the new distance from the tip by adding its length to the distance of the successor:
            n.distance_from_tip = son.distance_from_tip + n.length
        except:
            # If there is no successor because the element is an apex or a root nodule:
            # Then we simply define the distance to the tip as the length of the element:
            n.distance_from_tip = n.length

        
if __name__ == "__main__":
    scenarios = ms.from_table(file_path="inputs/Scenarios_24_09_22.xlsx", which=["Input_RSML_R4_D13"])

    for scenario_name, scenario in scenarios.items():
        g = scenario["input_mtg"]["root_mtg_file"]
        update_distance_from_tip(g)
        add_root_order_when_branching_is_wrong(g)


        plotter = pv.Plotter(off_screen=False, window_size=[1088, 1920], lighting="three lights")
        plotter.set_background("white")
        

        root_system_mesh, color_property, root_hair_mesh = plot_mtg_alt(g, cmap_property="root_order", flow_property=False)


        clim = [1, 2]

        plotter.add_mesh(root_system_mesh, cmap="jet", clim=clim, show_edges=False, log_scale=False)

        picker = VertexPicker(g=g, target_property="radius")
        picked_actor = plotter.enable_point_picking(callback=picker, picker='volume')

        plotter.reset_camera()
        plotter.show(interactive_update=False)