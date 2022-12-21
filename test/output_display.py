import openalea.plantgl.all as pgl
from rhizodep.tools import plot_mtg
import matplotlib.pyplot as plt
import numpy as np

def plot_N(g, range_min, range_max, p):

    if range_min == 0 and range_max == 0:
        props = g.property(p)
        max_scale = g.max_scale()
        plot_range = [props[vid] for vid in g.vertices(scale=max_scale) if props[vid] != 0]
        range_min, range_max = min(plot_range), max(plot_range)

        plt.ion()
        x = np.array([range_min, range_max])
        plt.pcolormesh([x, x], cmap='jet', vmin=range_min, vmax=range_max)
        plt.colorbar(location = 'top')
        plt.cla()
        plt.axis('off')


    scene = plot_mtg(g,
                     prop_cmap=p,
                     lognorm=False,  # to avoid issues with negative values
                     vmin=range_min,
                     vmax=range_max
                     )
    pgl.Viewer.display(scene)

    return range_min, range_max


def print_g(g, select, vertice):

    # extract MTG properties only once
    props = g.properties()
    extract = [props[k] for k in select]

    if vertice != 0:
        # print only selected segment
        print(vertice, end=' ')
        for k in range(len(extract)):
            print(select[k] + ' : ', end=' ')
            print(f"{extract[k][vertice]:4.15f}", end=' ')
        print('')

    else:
        max_scale = g.max_scale()
        for vid in g.vertices(scale=max_scale):
            # print for each segment selected properties in select
            print(vid, end=' ')
            for k in range(len(extract)):
                print(select[k] + ' : ', end=' ')
                print(f"{extract[k][vid]:4.15f}", end=' ')
            print('')
        print(props['xylem_Nm'][1], props['xylem_volume'][1])
