
from openalea.mtg.aml import MTG
from openalea.mtg.plantframe.dresser import dressing_data_from_file
from openalea.mtg.plantframe.plantframe import PlantFrame, compute_axes, build_scene
g = MTG('agraf.mtg')
dressing_data = dressing_data_from_file('agraf.drf')
topdia = lambda x:  g.property('TopDia').get(x)
pf = PlantFrame(g, TopDiameter=topdia,    DressingData = dressing_data)
axes = compute_axes(g, 3, pf.points, pf.origin)
diameters = pf.algo_diameter()
scene = build_scene(pf.g, pf.origin, axes, pf.points, diameters, 10000)
from openalea.plantgl.all import Viewer
Viewer.display(scene)
